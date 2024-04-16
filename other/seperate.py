import sys

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
import time
import datetime
from pathlib import Path

from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.musicldm import MusicLDM

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
from utilities.data.dataset import AudiostockDataset

from latent_diffusion.util import instantiate_from_config
    



from latent_diffusion.models.ddim import DDIMSampler
from typing import *
from tqdm import tqdm
import torch.nn as nn
import IPython.display as ipd
from  typing import *
import random

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os
import datetime
import soundfile as sf


# Assuming argparse is used to get the function argument from outside
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--function', type=str, choices=['differential_with_dirac', 'differential_with_gaussian'],
                    required=True, help='The differential function to use')
parser.add_argument('--s_churn', type=float, required=False, default =20.0)
# parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--config', type=str, required=True)

args = parser.parse_args()


s_churn = args.s_churn
function_name = args.function
# gpu = args.gpu #"cuda:0"
config = args.config




config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

# seed_everything(0)
batch_size = config["data"]["params"]["batch_size"]
log_path = config["log_directory"]
os.makedirs(log_path, exist_ok=True)

data = instantiate_from_config(config["data"])
data.prepare_data()
data.setup()


gpu = "cuda:"+str(config["trainer"]["devices"][0])

latent_diffusion = MusicLDM(**config["model"]["params"]).to(gpu)



class Schedule(nn.Module):
    """Interface used by different schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()
    
class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas

# ==================================================================================================================


def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    b, s, _, h, w = x.shape
    x_reshape = x 
    num_sources = s 

    # # focusing on one source 0, creating a differential emphasis on this source within the mixture.
    x_reshape[:, source_id, :,:] = mixture - (x_reshape.sum(dim=1)/4.0 - x_reshape[:, source_id, :,:,:])
    x = x_reshape 

    score = denoise_fn(x/torch.sqrt(sigma**2 + 1), sigma)
    
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    ds = torch.stack(ds, dim=1)

    return ds


def differential_with_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=lambda s: 0.75 * s):
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    # d = (x - denoise_fn(x, sigma=sigma)) / sigma 
    d = denoise_fn(x/torch.sqrt(sigma**2 + 1), sigma)

    b, s, _, h, w = x.shape
    x_reshape = x  

    d_reshape = d

    # source_id = random.randint(0, 3)
    for source_id in range(4):
        d_reshape[:,source_id] = d_reshape[:,source_id] - sigma / (2 * gamma ** 2) * (mixture - x_reshape.sum(dim=1)/4.) 
        #d = d - 8/sigma * (mixture - x.sum(dim=[1], keepdim=True)) 
        # d_reshape[:,source_id] = d_reshape[:,source_id] - 8 / sigma * (mixture - x_reshape.sum(dim=1)) 

    d = d_reshape
    return d

@torch.no_grad()
def separate_mixture(
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    noises: Optional[torch.Tensor],
    differential_fn: Callable = differential_with_dirac,
    s_churn: float = 0.0, # > 0 to add randomness
    num_resamples: int = 1,
    use_tqdm: bool = False,
):      
    # Set initial noise
    x = sigmas[0] * noises # [batch_size, num-sources, sample-length]
    mixture = sigmas[0] * mixture 
    
    vis_wrapper  = tqdm if use_tqdm else lambda x:x 
    for i in vis_wrapper(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i+1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # Compute conditioned derivative
            if random.random() < 0.00:
                d = denoise_fn(x/torch.sqrt(sigma_hat**2 + 1), sigma_hat)
            else:
                d = differential_fn(mixture=mixture, x=x, sigma=sigma_hat, denoise_fn=denoise_fn)
                

            # Update integral
            x = x + d * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + torch.sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)
    
    return x



def from_karras_to_t_adapter(sigma, karras_noise_levels, total_steps=1000):
    t = (total_steps-1) * (sigma/torch.max(karras_noise_levels))
    return t.long()





@torch.no_grad()
def denoise_fn(x, sigma):
    # t = sigma
    t = from_karras_to_t_adapter(sigma, karras_noise_levels)
    # print(t)
    b = x.shape[0]
    t = torch.full((b,), t, device= x.device, dtype=torch.long)
    return denoise_model.model.apply_model(x, t, cond = None)


#==========================================================================================================


if function_name == "differential_with_gaussian":
    func = differential_with_gaussian
if function_name == "differential_with_dirac":
    func = differential_with_dirac



# Generation hyper-parameters
s_churn = s_churn
batch_size = batch_size
num_steps = 200
total_steps = 1000
source_id = 0 
num_resamples = 2
device = gpu


sigma_schedule = KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=1.0)

karras_noise_levels = sigma_schedule(num_steps, device = device)


denoise_model = DDIMSampler(latent_diffusion.to(device))
denoise_model.make_schedule(  ddim_num_steps=num_steps, ddim_eta=1.0, verbose=False) 
timesteps = torch.tensor(np.flip(denoise_model.ddim_timesteps).copy())
 
# Creating the base directory if it doesn't exist
base_dir = "/home/karchkhadze/MusicLDM-Ext/lightning_logs/multichannel_slakh"

# Current date and time
now = datetime.datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

file_name = f"{formatted_now}_{function_name}_{s_churn}"
log_dir = os.path.join(base_dir, file_name)
os.makedirs(log_dir, exist_ok=True)


gen_dir = os.path.join(log_dir, "generated")
os.makedirs(gen_dir, exist_ok=True)
orig_dir = os.path.join(log_dir, "original")
os.makedirs(orig_dir, exist_ok=True)


# instruments
inst = ['bass', 'drums', 'guitar', 'piano']

# Initialize a counter before the loop starts
folder_counter = 0

for i, batch in enumerate(data.val_dataloader()):
    mixture = batch['fbank']
    mixture_z, _ = latent_diffusion.get_input(batch, 'fbank' )

    original_audios = batch["waveform_stems"]



    img = separate_mixture(
        mixture=mixture_z,
        denoise_fn=denoise_fn,
        # sigmas=timesteps,
        sigmas = karras_noise_levels,
        noises=torch.randn(batch_size, 4, 8, 256, 16).to(device),
        differential_fn = func,
        s_churn = s_churn, # > 0 to add randomness
        num_resamples = num_resamples,
        use_tqdm = True,
        )


    samples = latent_diffusion.adapt_latent_for_VAE_decoder(img)
    mel = latent_diffusion.decode_first_stage(samples)

    waveform = latent_diffusion.mel_spectrogram_to_waveform(mel, save=False)

    bs, _, _ = original_audios.shape
    for j in range(bs):
        # Use the counter for folder naming
        batch_gen_dir = os.path.join(gen_dir, str(folder_counter))
        batch_orig_dir = os.path.join(orig_dir, str(folder_counter))
        os.makedirs(batch_gen_dir, exist_ok=True)
        os.makedirs(batch_orig_dir, exist_ok=True)

        # Save original audio files
        for k in range(4):
            sf.write(os.path.join(batch_orig_dir, f"{inst[k]}.wav"), original_audios[j, k].cpu().numpy(), samplerate=16000)

        # Save generated audio files
        for k in range(4):
            sf.write(os.path.join(batch_gen_dir, f"{inst[k]}.wav"), waveform[j*4 + k].squeeze(), samplerate=16000)

        # Increment the folder counter after processing each sample
        folder_counter += 1


    if folder_counter >=500:
        break

    



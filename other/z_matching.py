import sys
import wandb

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
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
from utilities.data.dataset import AudiostockDataset

from latent_diffusion.util import instantiate_from_config
    

config = yaml.load(open("config/multichannel_LDM/multichannel_musicldm_slakh_uncond_test_inpaint.yaml", 'r'), Loader=yaml.FullLoader)

# seed_everything(0)
batch_size = config["data"]["params"]["batch_size"]
log_path = config["log_directory"]


print(f'Batch Size {batch_size} | Log Folder {log_path}')

data = instantiate_from_config(config["data"])
data.prepare_data()
data.setup()

latent_diffusion = MusicLDM(**config["model"]["params"]) #to("cuda:0")





#=============================================================================================================================



import matplotlib.pyplot as plt

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from latent_diffusion.models.ddim import DDIMSampler


from tqdm import tqdm
from latent_diffusion.modules.diffusionmodules.util import noise_like
import IPython.display as ipd
from torch.utils.checkpoint import checkpoint
import soundfile as sf
from pytorch_lightning.loggers import WandbLogger
import torchaudio



log_project = os.path.join(log_path, "z_matching")
os.makedirs(log_project, exist_ok=True)

# Initialize wandb logging
wandb_logger = WandbLogger(project="z_matching", log_model=False, save_dir=log_project)


# # Manually initialize wandb
# wandb.init(project="z_matching", dir=log_project, mode="online")

# Now set up PyTorch Lightning with the existing wandb run
# wandb_logger = WandbLogger(project="z_matching", log_model=False, experiment=wandb.run)


class z_matching(pl.LightningModule):
    def __init__(self, batch_size, model, learning_rate=0.01, dataset=None):
        super().__init__()
        # self.save_hyperparameters()
        self.vector = nn.Parameter(torch.randn(batch_size,32,256,16))  # Trainable vector
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset

        # Example of a frozen sub-network (let's use a simple linear layer)
        self.frozen_network = model

        self.ddim_steps = 20
        self.temperature = 1.0

        self.mel_transform = self.setup_mel_transform()



    def separate(self):
        # Forward pass through the frozen network

        b =self.vector.shape[0]
        cond = None
        shape = (b,self.frozen_network.channels, self.frozen_network.latent_t_size, self.frozen_network.latent_f_size) 
        mask=None
        x0=None
        # self.temperature=1.0
        unconditional_guidance_scale=1.0
        unconditional_conditioning=None
        repeat_noise=False

        x_T = self.vector
        device = self.device

        with self.frozen_network.ema_scope("Plotting"): 
            with torch.no_grad():

                if x_T is None:
                    img = torch.randn(shape, device=device)
                else:
                    img = x_T


                verbose=False
                eta=1.0

                self.ddim_sampler = DDIMSampler(self.frozen_network)
                self.ddim_sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=eta, verbose=verbose) # ddim_discretize = "quad",

                self.alphas = self.ddim_sampler.ddim_alphas
                self.alphas_prev = self.ddim_sampler.ddim_alphas_prev
                self.sqrt_one_minus_alphas =  self.ddim_sampler.ddim_sqrt_one_minus_alphas
                self.sigmas = self.ddim_sampler.ddim_sigmas


                timesteps = self.ddim_sampler.ddim_timesteps
                time_range = np.flip(timesteps)
                total_steps = timesteps.shape[0]

                print(f"Running DDIM Sampling with {total_steps} timesteps")

                iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    t = torch.full((b,), step, device=device, dtype=torch.long)
                    if mask is not None:
                        assert x0 is not None
                        img_orig = self.ddim_sampler.model.q_sample(
                            x0, t
                        )  # TODO deterministic forward pass?
                        img = (
                            img_orig * mask + (1.0 - mask) * img
                        )  # In the first sampling step, img is pure gaussian noise

            ###################################################################################################

                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                        e_t = self.ddim_sampler.model.apply_model(img, t, cond)
                    else:
                        x_in = torch.cat([img] * 2)
                        t_in = torch.cat([t] * 2)
                        c_in = torch.cat([unconditional_conditioning, cond])
                        e_t_uncond, e_t = self.ddim_sampler.model.apply_model(x_in, t_in, c_in).chunk(2)
                        # When unconditional_guidance_scale == 1: only e_t
                        # When unconditional_guidance_scale == 0: only unconditional
                        # When unconditional_guidance_scale > 1: add more unconditional guidance
                        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)


                    # select parameters corresponding to the currently considered timestep
                    a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=device)
                    a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[index], device=device)
                    sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=device)
                    # sigma_t = torch.full((b, 1, 1, 1), 0.0, device=device)
                    sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[index], device=device)

                    # current prediction for x_0
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device, repeat_noise) * self.temperature
                    img = a_prev.sqrt() * pred_x0 + dir_xt + noise  # TODO


            #################################################################################################
                    img, pred_x0 = img, pred_x0



        print("generated vector shape:",img.shape)

        with torch.no_grad():
            
            samples = self.frozen_network.adapt_latent_for_VAE_decoder(img)


            # Decode fisrt stage
            # mel = self.frozen_network.decode_first_stage(samples)
            z = 1.0 / self.frozen_network.scale_factor * samples
            mel = self.frozen_network.first_stage_model.decode(z)

            # Vocode
            # waveform = latent_diffusion.mel_spectrogram_to_waveform(mel,  save=False)
            mel = mel.squeeze(1)
            mel = mel.permute(0, 2, 1)
            waveform = self.frozen_network.first_stage_model.vocoder(mel)
            waveform = waveform.cpu().detach().numpy()

            # Reshape the array to (2, 4, 163872)
            reshaped_waveform = waveform.reshape((b, 4, 163872))
        return reshaped_waveform


    def compute_step(self, img, t, cond, b, index, mask, unconditional_conditioning, unconditional_guidance_scale, repeat_noise, temperature):
        """This function will be used in the checkpoint call and should contain
           the computation you want to checkpoint."""

        if mask is not None:
            assert x0 is not None
            img_orig = self.ddim_sampler.model.q_sample(
                x0, t
            )  # TODO deterministic forward pass?
            img = (
                img_orig * mask + (1.0 - mask) * img
            )  # In the first sampling step, img is pure gaussian noise

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.ddim_sampler.model.apply_model(img, t, cond)

        else:
            x_in = torch.cat([img] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            e_t_uncond, e_t = self.ddim_sampler.model.apply_model(x_in, t_in, c_in).chunk(2)
            # When unconditional_guidance_scale == 1: only e_t
            # When unconditional_guidance_scale == 0: only unconditional
            # When unconditional_guidance_scale > 1: add more unconditional guidance
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)


        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=self.device)
        # sigma_t = torch.full((b, 1, 1, 1), 0.0, device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[index], device=self.device)

        # current prediction for x_0
        pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(img.shape, self.device, repeat_noise) * temperature
        img = a_prev.sqrt() * pred_x0 + dir_xt + noise  # TODO

        img, pred_x0 = img, pred_x0
        return img



    def forward(self):
        # Forward pass through the frozen network

        b =self.vector.shape[0]
        cond = None
        shape = (b,self.frozen_network.channels, self.frozen_network.latent_t_size, self.frozen_network.latent_f_size) 
        mask=None
        x0=None
        # self.temperature=0.0
        unconditional_guidance_scale=1.0
        unconditional_conditioning=None
        repeat_noise=False

        x_T = self.vector
        device = self.device

        with self.frozen_network.ema_scope("Plotting"): 
            # with torch.no_grad():

            if x_T is None:
                img = torch.randn(shape, device=device)
            else:
                img = x_T


            verbose=False
            eta=1.0

            self.ddim_sampler = DDIMSampler(self.frozen_network)
            self.ddim_sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=eta, verbose=verbose) # ddim_discretize = "quad",

            self.alphas = self.ddim_sampler.ddim_alphas
            self.alphas_prev = self.ddim_sampler.ddim_alphas_prev
            self.sqrt_one_minus_alphas =  self.ddim_sampler.ddim_sqrt_one_minus_alphas
            self.sigmas = self.ddim_sampler.ddim_sigmas



            timesteps = self.ddim_sampler.ddim_timesteps
            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]

            print(f"Running DDIM Sampling with {total_steps} timesteps")

            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                t = torch.full((b,), step, device=device, dtype=torch.long)
                img = checkpoint(self.compute_step, img, t, cond, b, index, mask, unconditional_conditioning, unconditional_guidance_scale, repeat_noise, self.temperature)


        print("generated vector shape:",img.shape)
            
        samples = self.frozen_network.adapt_latent_for_VAE_decoder(img)


        # Decode fisrt stage
        # mel = self.frozen_network.decode_first_stage(samples)
        z = 1.0 / self.frozen_network.scale_factor * samples
        mel = self.frozen_network.first_stage_model.decode(z)

        # Vocode
        # waveform = latent_diffusion.mel_spectrogram_to_waveform(mel,  save=False)
        mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.frozen_network.first_stage_model.vocoder(mel)
        # waveform = waveform.cpu().detach().numpy()

        # Reshape the array to (2, 4, 163872)
        reshaped_array = waveform.reshape((b, 4, 163872))
        # Sum along axis 1
        summed_waveform = reshaped_array.sum(axis=1)


        return summed_waveform

    # def configure_optimizers(self):
    #     # Only optimize the trainable parameters
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        # Explicitly optimize only self.vector
        optimizer = torch.optim.Adam([self.vector], lr=self.learning_rate)
        return optimizer

    def on_train_start(self):
        """Hook called before the training starts, used to log the original track."""
        batch = next(iter(self.train_dataloader()))
        y = batch[0][0]
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()  # Ensure the tensor is on CPU and convert to NumPy
        for i in range(min(y.shape[0], 4)):  # Log up to four samples from the batch
            sample_rate = 16000  # Assuming the sample rate is 16000Hz
            wandb.log({f"Original Audio {i}": wandb.Audio(y[i], sample_rate=sample_rate, caption=f"Original Sample {i}")})

        # for logging the first before training auduos
        self.on_train_epoch_end()             

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

    def on_train_epoch_end(self, unused=None):
        """This hook is automatically called at the end of each training epoch."""
        if self.current_epoch % 10 == 0:
            waveform = self.separate()
            log_dict = {}

            
            for i in range(min(4, waveform.shape[1])):  # Assumes waveform shape [batch, channels, length]
                audio_clip = waveform[0][i]
                sample_rate = 16000  # Assume sample rate is known
                log_dict [f"Audio Sample {i}"] = wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"Sample {i}")
                # wandb.log({f"Audio Sample {i}": wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"Sample {i}")})
            # print("==================")
            new_mix = waveform.sum(1)
            log_dict [f"New mixture"] = wandb.Audio(new_mix[0], sample_rate=sample_rate, caption=f"new mixture")
            # wandb.log({f"New mixture": wandb.Audio(new_mix[0], sample_rate=sample_rate, caption=f"new mixture")})

            wandb.log(log_dict)

    def setup_mel_transform(self):
        window_size = 1024
        hop_size = 160
        n_fft = 1024  # Typically, n_fft is the same as window size
        n_mels = 128  # Number of Mel bands

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=window_size,
            hop_length=hop_size,
            n_mels=n_mels,
            power=2.0,  # Using power spectrum (not magnitude)
        )
        return mel_transform

    def get_mel_from_waveform(self, waveform):
        # Ensure the waveform tensor is in the correct shape: (batch_size, 1, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if it's missing
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # Add batch dimension if it's missing

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        return mel_spec

    def loudness_penalty(self, y_hat, y):
        eps = 1e-6  # To avoid log of zero
        y_hat_power = torch.mean(y_hat**2, dim=1) + eps
        y_power = torch.mean(y**2, dim=1) + eps

        # Compute the MSE between the log power of y_hat and y
        penalty = torch.mean((torch.log(y_hat_power) - torch.log(y_power))**2)
        return penalty

    def training_step(self, batch, batch_idx):
        

        y = batch[0][0]
        y_hat = self()
        y_hat = y_hat[:,:163840]

        # Compute mel-spectrograms
        y_mel = self.get_mel_from_waveform(y)
        y_hat_mel = self.get_mel_from_waveform(y_hat)


        # Compute MSE loss on waveform
        waveform_loss = nn.functional.mse_loss(y_hat, y)
        # Compute MSE loss on mel-spectrogram for perceptual loss
        mel_loss = nn.functional.mse_loss(y_hat_mel, y_mel)

        loudness_loss = self.loudness_penalty(y_hat, y)

        # Combine losses
        loss =  waveform_loss + 0.1* loudness_loss# mel_loss   #  #+# You can weight these components differently if desired

        # print(self.frozen_network.model.diffusion_model.input_blocks[0][0].weight[0][0])
        # print(self.vector[0][0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.current_epoch % 5 == 0:
        log_dict = {}

        new_mix = y_hat.cpu().detach().numpy()
        log_dict [f"train mixture"] = wandb.Audio(new_mix[0], sample_rate=16000, caption=f"train mixture")
        # target_mix = y.cpu().detach().numpy()
        # log_dict [f"target mixture"] = wandb.Audio(target_mix[0], sample_rate=16000, caption=f"target mixture")
        wandb.log(log_dict)

        return loss



batch_size = 1

# batch = next(iter(data.train_dataloader()))
# mixture = batch["waveform"][0].unsqueeze(0)

mixture, _ = torchaudio.load("lightning_logs/z_matching/wandb/run-20240421_220831-jiiwktg0/files/media/audio/Original Audio 0_0_ec25210b671ae6869910.wav")

mixture = mixture.repeat(10, 1, 1)


# Data
# Dummy dataset and dataloader for example
dataset = torch.utils.data.TensorDataset(mixture)


# Model
model = z_matching(batch_size=batch_size, model = latent_diffusion , dataset=dataset) #.to("cuda:0")


# Specify the path to your checkpoint
checkpoint_path = None #"/home/karchkhadze/MusicLDM-Ext/lightning_logs/z_matching/z_matching/emur7c9k/checkpoints/epoch=53-step=540.ckpt"

# # Train
trainer = Trainer(max_epochs=10000, 
                logger=wandb_logger,
                accelerator="gpu", 
                devices = [0],
                    # default_root_dir=default_root_dir,
                log_every_n_steps=1,
                resume_from_checkpoint=checkpoint_path,
                )

trainer.fit(model)


# waveform = model.separate()


# for i in range(4):
#     print(f" new mixture {i}")
#     # ipd.display(ipd.Audio(waveform[0][i], rate=16000))
#     sf.write(os.path.join(log_project, str(i)+".wav"), waveform[0][i], 16000)

# print("==================")
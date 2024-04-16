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
    

config = yaml.load(open("config/multichannel_LDM/multichannel_musicldm_slakh_uncond_3d_eval_seperation.yaml", 'r'), Loader=yaml.FullLoader)

# seed_everything(0)
batch_size = config["data"]["params"]["batch_size"]
log_path = config["log_directory"]
os.makedirs(log_path, exist_ok=True)

print(f'Batch Size {batch_size} | Log Folder {log_path}')

data = instantiate_from_config(config["data"])
data.prepare_data()
data.setup()

latent_diffusion = MusicLDM(**config["model"]["params"]).to("cuda:1")


from collections import defaultdict
from typing import *
import torchaudio
import torch
from pathlib import Path
import IPython.display as ipd
import soundfile as sf
from tqdm import tqdm


def load_chunks(chunk_folder: Path, stems: Sequence[str]) -> Tuple[Mapping[str, torch.Tensor], int]:
    separated_tracks_and_rate = {s: torchaudio.load(chunk_folder / f"{s}.wav") for s in stems}
    separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
    sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

    assert len({*sample_rates_sep}) == 1, print(sample_rates_sep)
    sr = sample_rates_sep[0]

    return separated_tracks, sr


def latent_to_waveform(img):

    # samples = latent_diffusion.adapt_latent_for_VAE_decoder(img)
    mel = latent_diffusion.decode_first_stage(img)

    waveform = latent_diffusion.mel_spectrogram_to_waveform(mel, save=False)
    # waveform = np.nan_to_num(waveform)
    # waveform = np.clip(waveform, -1, 1)
    return waveform

def waveform_to_latent(waveform):
    # Placeholder list to store each processed mel spectrogram
    # Process each waveform to get its mel spectrogram
    mel = data.train_dataset.get_mel_from_waveform(waveform.numpy()[0])


    # # Stack the list of tensors to get a batch again
    # # The unsqueeze adds an additional dimension to match your desired shape for further processing
    # mel = torch.stack(mel) #.unsqueeze(1)


    fake_batch = {}
    fake_batch['fbank'] = torch.tensor(mel).unsqueeze(0)
    z, _ = latent_diffusion.get_input(fake_batch, 'fbank' )
    return z

# Update directory paths
base_dir = "/home/karchkhadze/MusicLDM-Ext/lightning_logs/multichannel_slakh/2024-04-09_01-42-19_differential_with_gaussian_20.0/"

separation_dir = base_dir + "generated"
dataset_path = base_dir + "original"
original_dir_VAE_HiFi = base_dir + "original_VAE_HiFi"
mixture_dir_VAE_HiFi =  base_dir + "mixture_VAE_HiFi"
os.makedirs(original_dir_VAE_HiFi, exist_ok=True)
os.makedirs(mixture_dir_VAE_HiFi, exist_ok=True)

separation_path = Path(separation_dir)
dataset_path = Path(dataset_path)
original_dir_VAE_HiFi =Path(original_dir_VAE_HiFi)
mixture_dir_VAE_HiFi = Path(mixture_dir_VAE_HiFi)

stems = ["bass","drums","guitar","piano"]

df_entries = defaultdict(list)
# Loop through folders
for folder_id in tqdm(range(504), desc="Processing folders"):

    # load seperated tracks and resample track
    separated_track, _ = load_chunks(separation_path/str(folder_id), stems)

    # load original track
    original_track, _ = load_chunks(dataset_path/str(folder_id), stems)

    # separated_track  = original_track

    # Compute mixture
    mixture = sum([owav for owav in original_track.values()])

    # compute VAE+hHiFi gan for each stem
    for k in original_track:
        z = waveform_to_latent(original_track[k])
        separated_track_VAE_HiFi = latent_to_waveform(z)
        # print()

        # Assuming you want to save separated tracks
        # Save the separated track in the corresponding directory with name k in folder number i
        output_dir = original_dir_VAE_HiFi/str(folder_id)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{k}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, separated_track_VAE_HiFi.squeeze(), 16000)


    # Process mixture through latent space transformation
    z_mixture = waveform_to_latent(mixture)
    mixture_VAE_HiFi = latent_to_waveform(z_mixture)

    # Save mixtures
    output_dir = mixture_dir_VAE_HiFi / str(folder_id)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "mixture.wav"  # You can change the filename if needed
    output_path = output_dir / output_filename
    sf.write(output_path, mixture_VAE_HiFi.squeeze(), 16000)  # Assuming sample rate is 16000 Hz    



    # if folder_id>5:
    #     break


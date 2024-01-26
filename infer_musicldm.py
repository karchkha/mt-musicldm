'''
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
'''
import sys

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
import time

from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.musicldm import MusicLDM
from utilities.data.dataset import AudiostockDataset, TextDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
    

config_path = '/home/kechen/research/CTTM/Controllable_TTM/config/musicldm_audiostock10k/musicldm.yaml'

def main(config, texts, seed):
    seed_everything(seed)
    batch_size = config["model"]["params"]["batchsize"]

    log_path = os.path.join('/home/kechen/research/CTTM/Controllable_TTM/config/', os.getlogin())
    os.makedirs(log_path, exist_ok=True)
    folder_name = os.listdir(log_path)
    i = 0
    while str(i) in folder_name:
        i = i + 1
    log_path = os.path.join(log_path, str(i))
    os.makedirs(log_path, exist_ok=True)


    print(f'Samples with be saved at {log_path}')

    dataset = TextDataset(
        data = texts,
        logfile=os.path.join(log_path, "meta.txt")
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['model']['num_workers']
    )

    devices = torch.cuda.device_count()

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)
    trainer = Trainer(
        max_epochs=1200,
        accelerator="gpu",
        devices=devices,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (int(devices) > 1)
        else None,
    )

    trainer.validate(latent_diffusion, loader)
    

    print(f"Generation Finished. Please check the generation samples and the meta file at {log_path}")

def print_license():
    print('MusicLDM is is released and maintained by:')
    print('Ke Chen, Yusong Wu, Haohe Liu')
    print('with the paper: MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies')
    print('----------------------------------------------')
    print('MusicLDM is released under the Creative Common NonCommercial 4.0 License (CC BY-NC)')
    print('Please read this license carefully at https://creativecommons.org/licenses/by-nc/4.0/legalcode ')
    print('CC BY-NA lets you remix, adapt, and build upon MusicLDM non-commercially, new works must also acknowledge MusicLDM and be non-commercial.')
    print(f'As the user {os.getlogin()}, do you agree with this (yes/no)?')
    ans = input()
    if ans != 'yes':
        raise f'User does not agree with CC BY-NA, end MusicLDM'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="text to generate the music sample from",
        default=""
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=False,
        help="a path to text file to generate the music samples from",
        default=""
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="a generation seed",
        default=0
    )

    args = parser.parse_args()

    if args.text != "" and args.texts != "":
        raise f'********** Error: Only one of text and texts configuration could be given **********'
    
    if args.text != "":
        print(f'********** MusicLDM: generate music sample from the text:')
        print('----------------------------------------------')
        print(f'{args.text}')
        print('----------------------------------------------')
        print_license()
        texts = [args.text]
    
    if args.texts != "":
        print(f'********** MusicLDM: generate music samples from the text file:')
        print('----------------------------------------------')
        print(f'{args.texts}')
        print('----------------------------------------------')
        print_license()
        texts = np.genfromtxt(args.texts, dtype=str, delimiter="\n")

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    main(config, texts, args.seed)


import sys

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
import time

from utilities.data.dataset import AudiostockDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset

# from utilities.sampler import DistributedSamplerWrapper
# from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from latent_encoder.autoencoder import AutoencoderKL
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f

def get_restore_step(path):
    checkpoints = os.listdir(path)
    steps = [int(x.split(".ckpt")[0].split("step=")[1]) for x in checkpoints]
    return checkpoints[np.argmax(steps)], np.max(steps)


def main(config):
    batch_size = config["model"]["params"]["batchsize"]
    log_path = config["log_directory"]
    os.makedirs(log_path, exist_ok=True)

    print(f'Batch Size {batch_size} | Log Folder {log_path}')
    if config["path"]["dataset_type"] == 'audiostock':
        dataset = AudiostockDataset(
            dataset_path=config["path"]["train_data"],
            label_path=config["path"]["label_data"],
            config=config,
            train=True,
            factor=1.0
        )
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True
        )

        val_dataset = AudiostockDataset(
            dataset_path=config["path"]["test_data"],
            label_path=config["path"]["label_data"],
            config=config,
            train=False,
            factor=1.0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

        test_dataset = AudiostockDataset(
            dataset_path=config["path"]["test_data"],
            label_path=config["path"]["label_data"],
            config=config,
            train=False,
            factor=1.0,
            whole_track=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

    # No normalization here
    model = AutoencoderKL(
        config["model"]["params"]["ddconfig"],
        config["model"]["params"]["lossconfig"],
        embed_dim=config["model"]["params"]["embed_dim"],
        image_key=config["model"]["params"]["image_key"],
        base_learning_rate=config["model"]["base_learning_rate"],
        subband=config["model"]["params"]["subband"],
        config=config
    )

    config["id"]["version"] = "%s_%s_%s_%s_%s_%s" % (
        config["id"]["name"],
        config["model"]["params"]["embed_dim"],
        config["model"]["params"]["ddconfig"]["ch"],
        float(config["model"]["base_learning_rate"]),
        config["id"]["version"],
        int(time.time())
    )
    if config["path"]["test"]:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(
            save_dir=log_path,
            version=config["id"]["version"],
            project=config["project_name"],
            config=config,
            name=config["id"]["version"]
        )

    checkpoint_step_callback = ModelCheckpoint(
        monitor="train_step",
        mode="max",
        filename="checkpoint-{train_step:.0f}-{aeloss_val:.3f}",
        every_n_train_steps=config["step"]["save_checkpoint_every_n_training_batchs"] * 2,  
        # When you have two optimizer, one traditional step equals to two train steps.
        save_top_k=config["step"]["save_top_k"],
        save_last=True,
    )

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="aeloss_val",
        mode="min",
        filename="checkpoint-{train_step:.0f}-{aeloss_val:.3f}",
        every_n_train_steps=config["step"]["save_checkpoint_every_n_training_batchs"] * 2,  
        # When you have two optimizer, one traditional step equals to two train steps.
        save_top_k=config["step"]["save_top_k"],
        save_last=True,
    )

    checkpoint_path = os.path.join(
        log_path,
        config["project_name"],
        config["id"]["version"],
        "checkpoints",
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print("Train from scratch")
    devices = torch.cuda.device_count()
    # devices = 1

    trainer = Trainer(
        max_epochs=1000,
        default_root_dir=log_path,
        resume_from_checkpoint=None,
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_step_callback, checkpoint_loss_callback],
        # check_val_every_n_epoch=10,
        val_check_interval=config["step"]["val_check_interval"],
        num_sanity_val_steps=10
    )
    print(config["path"]["test"])
    if config["path"]["test"]:
        # EVALUTION
        ckpt_path = config["path"]["ckpt_path"]
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        for n,p in model.named_parameters():
            if n in ckpt:
                print(n, "Loaded")
            else:
                print(n, "Unloaded")
        model.load_state_dict(ckpt, strict=False)
        trainer.test(model, test_loader)
    else:
        # TRAINING
        trainer.fit(model, loader, val_loader)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to autoencoder config",
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)

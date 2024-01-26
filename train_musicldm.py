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
from utilities.data.dataset import AudiostockDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
    
    

def main(config):
    seed_everything(0)
    batch_size = config["model"]["params"]["batchsize"]
    log_path = config["log_directory"]
    os.makedirs(log_path, exist_ok=True)

    print(f'Batch Size {batch_size} | Log Folder {log_path}')

    if config['path']['dataset_type'] == 'audiostock':
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
            num_workers=config['model']['num_workers'],
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
            num_workers=config['model']['num_workers']
        )
    
    config["id"]["version"] = "%s_%s_%s_%s" % (
        config["id"]["name"],
        float(config["model"]['params']["base_learning_rate"]),
        config["id"]["version"],
        int(time.time())
    )

    wandb_logger = WandbLogger(
        save_dir=log_path,
        version=config["id"]["version"],
        project=config["project_name"],
        config=config,
        name=config["id"]["version"]
    )

    try:
        config_reload_from_ckpt = config["model"]["params"]["ckpt_path"]
    except:
        config_reload_from_ckpt = None

    validation_every_n_steps = config["step"]["validation_every_n_steps"]
    save_checkpoint_every_n_steps = config["step"][
        "save_checkpoint_every_n_steps"
    ]
    save_top_k = config["step"]["save_top_k"]

    if validation_every_n_steps > len(loader):
        validation_every_n_epochs = int(validation_every_n_steps / len(loader))
        validation_every_n_steps = None
    else:
        validation_every_n_epochs = None

    assert not (
        validation_every_n_steps is not None and validation_every_n_epochs is not None
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )

    checkpoint_path = os.path.join(
        log_path,
        config["project_name"],
        config["id"]["version"],
        "checkpoints",
    )

    os.makedirs(checkpoint_path, exist_ok=True)

    if len(os.listdir(checkpoint_path)) > 0:
        print("++ Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.test_data_subset_path = config['path']['test_data']
    trainer = Trainer(
        max_epochs=1200,
        accelerator="gpu",
        devices=devices,
        num_sanity_val_steps=0,
        # resume_from_checkpoint=resume_from_checkpoint,
        logger=wandb_logger,
        #   limit_val_batches=2,
        val_check_interval=validation_every_n_steps,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (int(devices) > 1)
        else None,
        callbacks=[checkpoint_callback],
    )
    if config['test_mode']:
        # Evaluation / Validation
        trainer.validate(latent_diffusion, val_loader)
    else:
        # Training
        trainer.fit(latent_diffusion, loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to musicldm config",
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)


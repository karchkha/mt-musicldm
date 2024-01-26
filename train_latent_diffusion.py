import sys

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.ddpm import LatentDiffusion
from utilities.data.dataset import Dataset as AudioSetDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
    
    

def main(configs):
    seed_everything(0)

    preprocess_config, train_config, latent_diffusion_config = configs
    log_path = latent_diffusion_config["log_directory"]
    batch_size = latent_diffusion_config["model"]["params"]["batchsize"]
    test_data_subset_folder = preprocess_config["path"]["test_data_folder"]
    
    copy_test_subset_data(preprocess_config["path"]["test_data"], test_data_subset_folder)
    
    if train_config["augmentation"]["balanced_sampling"]:
        print("balanced sampler is being used")
        samples_weight = np.loadtxt(
            preprocess_config["path"]["train_data"][:-5] + "_weight.csv", delimiter=","
        )
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )

        try:
            mixup_no_bal = train_config["augmentation"]["mixup_no_bal"]
        except Exception as e:
            print(e)
            mixup_no_bal = False

        if mixup_no_bal:
            samples_weight = None

        dataset = AudioSetDataset(
            preprocess_config, train_config, samples_weight=samples_weight, train=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=16,
            pin_memory=True,
        )
    else:
        print("balanced sampler is not used")
        dataset = AudioSetDataset(preprocess_config, train_config, train=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=True,
        )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    # Get dataset
    val_dataset = AudioSetDataset(preprocess_config, train_config, train=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
    )

    latent_diffusion_config["id"]["version"] = "%s_%s" % (
        latent_diffusion_config["id"]["name"],
        latent_diffusion_config["id"]["version"],
    )

    wandb_logger = WandbLogger(
        save_dir=log_path,
        version=latent_diffusion_config["id"]["version"],
        project=train_config["project_name"],
        config={**preprocess_config, **train_config, **latent_diffusion_config},
        name=latent_diffusion_config["id"]["name"],
    )

    try:
        config_reload_from_ckpt = latent_diffusion_config["model"]["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    validation_every_n_steps = train_config["step"]["validation_every_n_steps"]
    save_checkpoint_every_n_steps = train_config["step"][
        "save_checkpoint_every_n_steps"
    ]
    save_top_k = train_config["step"]["save_top_k"]

    if validation_every_n_steps > len(loader):
        validation_every_n_epochs = int(validation_every_n_steps / len(loader))
        validation_every_n_steps = None
    else:
        validation_every_n_epochs = None

    # if(save_top_k != -1 and len(loader) < 20000):
    #     print("++ Save checkpoint every one epoch")
    #     save_checkpoint_every_n_epochs = 1
    #     save_checkpoint_every_n_steps = None
    # else:
    #     save_checkpoint_every_n_epochs = None

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
        train_config["project_name"],
        latent_diffusion_config["id"]["version"],
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

    latent_diffusion = LatentDiffusion(**latent_diffusion_config["model"]["params"])
    latent_diffusion.test_data_subset_path = test_data_subset_folder
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        #   num_sanity_val_steps=0,
        resume_from_checkpoint=resume_from_checkpoint,
        logger=wandb_logger,
        #   limit_val_batches=2,
        val_check_interval=validation_every_n_steps,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (int(devices) > 1)
        else None,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_root", type=str, required=False, help="path to config folder"
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    config_root = args.config_root

    preprocess_config = os.path.join(config_root, "preprocess.yaml")
    train_config = os.path.join(config_root, "train.yaml")
    latent_diffusion_config = os.path.join(config_root, "latent_diffusion.yaml")

    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
    latent_diffusion_config = yaml.load(
        open(latent_diffusion_config, "r"), Loader=yaml.FullLoader
    )

    configs = (preprocess_config, train_config, latent_diffusion_config)

    main(configs)

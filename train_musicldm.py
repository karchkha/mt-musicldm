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
from utilities.data.dataset import AudiostockDataset, DS_10283_2325_Dataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data

# from latent_diffusion.util import instantiate_from_config
    
    

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
    elif config['path']['dataset_type'] == 'DS_10283_2325':
        dataset = DS_10283_2325_Dataset(
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

        val_dataset = DS_10283_2325_Dataset(
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

    # data = instantiate_from_config(config["data"])
    # data.prepare_data()
    # data.setup()

    # adding a random number of seconds so that exp folder names coincide less often
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')


    nowname = "%s_%s_%s_%s" % (
        now,
        config["id"]["name"],
        float(config["model"]['params']["base_learning_rate"]),
        config["id"]["version"],
        # int(time.time())
    )

    resume_from_checkpoint = config["trainer"]["resume_from_checkpoint"]

    assert not (config.get("trainer", {}).get("resume_from_checkpoint") is not None and config.get("trainer", {}).get("resume") is not None), \
        "You can't define both 'resume_from_checkpoint' and 'resume'. You need to choose one, as 'resume_from_checkpoint' continues from " \
        "the checkpoint in the different project, while 'resume' is for training continuation in the same folder."
    
    if config["trainer"]["resume"]:
        if not os.path.exists(config["trainer"]["resume"]):
            raise ValueError('Cannot find {}'.format(config["trainer"]["resume"]))
        if os.path.isfile(config["trainer"]["resume"]):
            paths = config["trainer"]["resume"].split('/')
            idx = len(paths)-paths[::-1].index('lightning_logs')+2
            logdir = '/'.join(paths[:idx])
            ckpt = config["trainer"]["resume"]
        else:
            assert os.path.isdir(config["trainer"]["resume"]), config["trainer"]["resume"]
            logdir = config["trainer"]["resume"].rstrip('/')
            # ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')
            # ckpt = sorted(glob.glob(os.path.join(logdir, 'checkpoints', '*.ckpt')))[-1]
            if Path(os.path.join(logdir, 'checkpoints', 'last.ckpt')).exists():
                ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')
            else:
                ckpt = None #sorted(Path(logdir).glob('checkpoints/*.ckpt'))[-1]

        resume_from_checkpoint = ckpt
        # base_configs = sorted(glob.glob(os.path.join(logdir, 'configs/*.yaml')))
        # config.base = base_configs+config.base
        _tmp = logdir.split('/')
        nowname = _tmp[_tmp.index('lightning_logs')+2]
        if config["dev"]:
            nowname = "DEV_EXP" 
            # logdir = "./lightning_logs/DEV_EXP" 
    else:
        if config["dev"]:
            nowname = "DEV_EXP" 
            # logdir = "./lightning_logs/DEV_EXP" 
        # os.makedirs(logdir, exist_ok=True)

    print("\nName of the run is:", nowname, "\n")

    run_path = os.path.join(
        log_path,
        config["project_name"],
        nowname,
    )

    os.makedirs(run_path, exist_ok=True)


    wandb_logger = WandbLogger(
        save_dir=run_path,
        # version=nowname,
        project= config["project_name"],
        config=config,
        name=nowname
    )

    wandb_logger._project = ""  # prevent naming experiment nama 2 time in logginf vals

    try:
        config_reload_from_ckpt = config["model"]["params"]["ckpt_path"]
    except:
        config_reload_from_ckpt = None

    validation_every_n_steps = config["trainer"]["validation_every_n_steps"]
    save_checkpoint_every_n_steps = config["trainer"][
        "save_checkpoint_every_n_steps"
    ]
    save_top_k = config["trainer"]["save_top_k"]

    if validation_every_n_steps > len(loader):
        validation_every_n_epochs = int(validation_every_n_steps / len(loader))
        validation_every_n_steps = None
    else:
        validation_every_n_epochs = None

    assert not (
        validation_every_n_steps is not None and validation_every_n_epochs is not None
    )

    checkpoint_path = os.path.join(
        log_path,
        config["project_name"],
        nowname,
        "checkpoints",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath= checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}", #TODO :FAD = frechet_audio_distance, no?
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )


    os.makedirs(checkpoint_path, exist_ok=True)

    if resume_from_checkpoint is None:
        if len(os.listdir(checkpoint_path)) > 0:
            print("++ Load checkpoint from path: %s" % checkpoint_path)
            restore_step, n_step = get_restore_step(checkpoint_path)
            resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
            print("Resume from checkpoint", resume_from_checkpoint)
        elif config_reload_from_ckpt is not None:
            resume_from_checkpoint = config_reload_from_ckpt
            print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
            resume_from_checkpoint = None
        else:
            print("Train from scratch")
            resume_from_checkpoint = None

    # devices = torch.cuda.device_count()

    ######################## GPU management #################
    # default to ddp
    devices = config["trainer"]["devices"]
    accelerator = config["trainer"]["accelerator"]
    max_epochs = config["trainer"]["max_epochs"]
    limit_train_batches = config["trainer"]["limit_train_batches"]
    limit_val_batches = config["trainer"]["limit_val_batches"]

    print(f'Running on {accelerator} with devices: {devices}')


    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.test_data_subset_path = config['path']['test_data']
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_sanity_val_steps=0,
        resume_from_checkpoint=resume_from_checkpoint,
        logger=wandb_logger,
        limit_val_batches=limit_train_batches,
        limit_train_batches = limit_val_batches,
        val_check_interval=validation_every_n_steps,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (len(devices) > 1)
        else None,
        callbacks=[checkpoint_callback],
    )
    if config['mode'] in ["test", "validate"]:
        # Evaluation / Validation
        trainer.validate(latent_diffusion, val_loader)
    if config['mode'] == "validate_and_train":
        # Training
        trainer.validate(latent_diffusion, val_loader)
        trainer.fit(latent_diffusion, loader, val_loader)
    elif config['mode'] == "train":
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


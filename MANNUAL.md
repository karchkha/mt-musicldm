# How to Use MusicLDM at IRCAM Server


## Step 1: Login to the ircam server "reach2.ircam.fr" 

Before doing so, make sure you have an account to the ircam server and get to know how to login from the ircam ssh main server.

## Step 2 : Install Conda:
If you already installed conda before, skip this step, otherwise:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then, after installing you should make sure your conda environment is running on your bash


## Step 3: Create the environment from the yaml file:
```
conda env create -f /data/reach/musicldm/musicldm_environment.yml
``` 
Input "yes" or "y" when you face any branch choice.

## Step 4: Activate the environment:
```
conda activate musicldm_env
```

## Step 5: Run MusicLDM:
Locate at the MusicLDM interface:
```
cd /data/reach/musicldm/interface
```
Everytime when you run the interface, you need to agree with (by typing "yes") the CC BY-NA license.

MusicLDM supports two running methods.

You can generate a 10-sec music sample from a text by:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --text "Cute toy factory theme loop"
```

Or you can write a txt file, each line of which contains a sentence for generation. And you would be able to generate each sample for each line by:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt
```
Please check out /data/reach/musicldm/interface/sample_file.txt for example.

"CUDA_VISIBLE_DEVICES" indicates the GPU you want to use, you can use the below command to check the availbility of GPUs in the server:
```
nvidia-smi
```
Then you can indicate the GPU index (0, 1, or 2) you want to use for running MusicLDM.

You can also pass a "seed" in the running code, such as:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --text "Cute toy factory theme loop" --seed 1423
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 1423
```
When using different seeds, usually you will get different generation samples.

## Step 6: Check out the generation:
Usually, the generation of MusicLDM will be saved in the below folder:
```
/data/reach/musicldm/generation/
```
You will find your username folder under this path, and you can check the generation. Don't worry about the replacement of the new generation to the old generation. MusicLDM will save them by creating a new subfolder under your username folder. 

## Attention: When you finish your running
Please delete your username folder if you have moved your generation, to save the disk storage.


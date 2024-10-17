# Multi-Track MusicLDM: Towards Versatile Music Generation with Latent Diffusion Model

In this work, we address multi-track music generation challenge by extending the MusicLDM—a latent diffusion model for music—into a multi-track generative model. By learning the joint probability of tracks sharing a context, our model is capable of generating music across several tracks that correspond well to each other, either conditionally or unconditionally. Additionally, our model is capable of arrangement generation, where the model can generate any subset of tracks given the others (e.g., generating a piano track complementing given bass and drum tracks). 

Sound examples can be found at https://mt-musicldm.github.io


# Installation

To install Multi-Track MusicLDM, follow these steps:

Clone the repository to your local machine
```bash
$ git clone https://github.com/karchkha/mt-musicldm
```

To run the code in this repository, you will need python 3.9 

Navigate to the project directory and install the required dependencies

If you already installed conda before, skip this step, otherwise:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then, after installing you should make sure your conda environment is running on your bash


```
conda env create -f musicldm_env.yml
``` 


then 
```
conda activate musicldm_env
```


# Data

In this project, the Slakh2100 data is used.

Please follow the instructions for data download and set up given here:

https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md

# Training Multi-Track MusicLDM

After data and conda evn are intalled properlly, you will need to dowload components of MusicLDM that are used for Multi-track MusicLDM too. For this please 

```
# Download hifigan-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/hifigan-ckpt.ckpt

# Download vae-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/vae-ckpt.ckpt

# Download clap-ckpt.pt
wget https://zenodo.org/record/10643148/files/clap-ckpt.pt

# Download musicldm-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/musicldm-ckpt.ckpt   # this is original MusicLDM which we don't need!
```

After placing this in some directory and changing corresponding links in the config file, for the trainion of Multi-Track MusicLDM please run:

```
# For un-conditional:
python train_musicldm.py --config config/multichannel_LDM/multichannel_musicldm_slakh_uncond_3d.yaml

# For conditional:
python train_musicldm.py --config config/multichannel_LDM/multichannel_musicldm_slakh_with_CLAP_3d.yaml
```

# Checkpoints

Plase download checkpoints from:

```
# For un-conditional:
wget https://zenodo.org/records/13947715/files/2024-03-24T19-51-37_3_D_4_stems_slakh_uncond_ch%3D192_3e-05_.tar.gz?download=1

# For conditional:
wget https://zenodo.org/records/13947715/files/2024-03-25T00-55-31_3_D_4_stems_slakh_with_CALP_ch%3D192_3e-05_.tar.gz?download=1
```

# Inference


```
# For un-conditional:
python train_musicldm.py --config config/multichannel_LDM/multichannel_musicldm_slakh_uncond_3d_eval.yaml

# For conditional:
python train_musicldm.py --config config/multichannel_LDM/multichannel_musicldm_slakh_3d_with_CLAP_eval_TEXT.yaml

# Arrangement Generation:
config/multichannel_LDM/multichannel_musicldm_slakh_3d_with_CLAP_eval_inpaint.yaml
```




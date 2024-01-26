import shutil
import os
import random
import json
import torchaudio
import torch
from audioldm_eval import EvaluationHelper
import numpy as np

waveform_save_path = '/graft1/datasets/kechen/audioldm_s_output/generation'
test_data_path = '/graft1/datasets/kechen/audiostock-10k-16khz/test'
evaluator = EvaluationHelper(16000, 'cuda')
metrics = evaluator.main(waveform_save_path, test_data_path)

# filelist = os.listdir(waveform_save_path)

# for f in filelist:
#     print(f)
#     y, sr = torchaudio.load(os.path.join(waveform_save_path ,f))
    # print(y.shape, sr)
    # y = torch.sum(y, dim=0, keepdim=True) / 2
    # y = torchaudio.functional.resample(y, sr, 16000)
    # torchaudio.save(os.path.join(waveform_save_path, f.split(".")[0] + '.wav'), y, 16000)
    # os.remove(os.path.join(waveform_save_path, f))
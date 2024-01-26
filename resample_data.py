import torchaudio
import os
from tqdm import tqdm

target_sr = 16000
chunk_size = 120


# First option
# data_folder = '/trunk/datasets/niloofar/mixes_audio_train'
# output_folder = '/trunk/datasets/kechen/soundcloud-16k/train'

# filelist = os.listdir(data_folder)

# for f in tqdm(filelist):
#     fp = os.path.join(data_folder,f)
#     y, sr = torchaudio.load(fp)
#     ny = torchaudio.functional.resample(y, sr, target_sr)
#     for i, offset in enumerate(range(0, ny.size(-1) - target_sr * chunk_size, target_sr * chunk_size)):
#         op = os.path.join(output_folder, f.split('.')[0] + '_' + str(i) + '.wav')
#         torchaudio.save(op, ny[:, offset: offset + target_sr * chunk_size], target_sr)

# Second option
data_folder = '/trunk/datasets/niloofar/mixes_audio_test'
output_folder = '/trunk/datasets/kechen/soundcloud-16k/test'

filelist = os.listdir(data_folder)

for f in tqdm(filelist):
    audiolist = os.listdir(os.path.join(data_folder, f))
    for a in audiolist:
        fa = os.path.join(data_folder, f, a)
        fo = os.path.join(output_folder, f + '_' + a)
        y, sr = torchaudio.load(fa)
        ny = torchaudio.functional.resample(y, sr, target_sr)
        torchaudio.save(fo, ny, target_sr)
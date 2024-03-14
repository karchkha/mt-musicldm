# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import sys

sys.path.append(
    "/home/karchkhadze/MusicLDM-Ext/src"
)
import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import utilities.audio as Audio
import librosa
import os
import torchvision
import yaml
import pandas as pd
import omegaconf

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup

class TextDataset(Dataset):
    def __init__(self, data, logfile):
        super().__init__()
        self.data = data
        self.logfile = logfile
    def __getitem__(self, index):
        data_dict = {}
         # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['fbank'] = np.zeros((1024,64))
        data_dict['waveform'] = np.zeros((32000))
        data_dict['text'] = self.data[index]
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(f"{data_dict['fname']}: {data_dict['text']}")
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{data_dict['fname']}: {data_dict['text']}")
        return data_dict


    def __len__(self):
        return len(self.data)


class AudiostockDataset(Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__()

        self.train = train
        self.config = config

        # self.read_datafile(dataset_path, label_path, train)

        self.data = []
        if type(dataset_path) is str:
            self.data = self.read_datafile(dataset_path, label_path, train) 

        elif type(dataset_path) is list or type(dataset_path) is omegaconf.listconfig.ListConfig:
            for datapath in dataset_path:
                self.data +=  self.read_datafile(datapath, label_path, train) 
   
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.total_len = int(len(self.data) * factor)


        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        self.whole_track = whole_track


        try:
            self.segment_size = config["preprocessing"]["audio"]["segment_size"]
            self.target_length = int(self.segment_size / self.hopsize)
            self.segment_length = int(self.target_length * self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None
        
        if not train:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.return_all_wav = False
        if self.mixup > 0:
            self.tempo_map = np.load(config["path"]["tempo_map"], allow_pickle=True).item()
            self.tempo_folder = config["path"]["tempo_data"]
        
        if self.mixup > 1:
            self.return_all_wav = config["augmentation"]["return_all_wav"] 

        print("Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s" % (self.mixup, self.timem, self.freqm, self.use_blur))

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        print(f'| Audiostock Dataset Length:{len(self.data)} | Epoch Length: {self.total_len}')

    def read_datafile(self, dataset_path, label_path, train):
        data = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        if (not train) and len(data) > 2000:
            data_dict = {}
            filelist = [os.path.basename(f).split('.')[0].split('_') for f in self.data]
            for f,idx in filelist:
                if f not in data_dict:
                    data_dict[f] = int(idx)
                else:
                    data_dict[f] = max(int(idx), data_dict[f])
            data = [os.path.join(dataset_path, f'{k}_{data_dict[k] // 2}.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_0.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_{data_dict[k]}.wav') for k in data_dict.keys()]

        
        self.label = []
        if label_path is not None:
            for d in data:
                lp = os.path.join(label_path, os.path.basename(d).split('.')[0] + '.json')
                assert os.path.exists(lp), f'the label file {lp} does not exists.'
                self.label.append(lp)   

        return data 
                
    def random_segment_wav(self, x):
        wav_len = x.shape[-1]
        assert wav_len > 100, "Waveform is too short, %s" % wav_len
        if self.whole_track:
            return x
        if wav_len - self.segment_length > 0:
            if self.train:
                sta = random.randint(0, wav_len -self.segment_length)
            else:
                sta = (wav_len - self.segment_length) // 2
            x = x[:, sta: sta + self.segment_length]
        return x
    
    def normalize_wav(self, x):
        x = x[0]
        x = x - x.mean()
        x = x / (torch.max(x.abs()) + 1e-8)
        x = x * 0.5
        x = x.unsqueeze(0)
        return x

    def read_wav(self, filename):
        y, sr = torchaudio.load(filename)
        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_mel(self, filename, mix_filename = None):
        # mixup
        if mix_filename is None:
            y = self.read_wav(filename)
        else:
            # get name 
            anchor_name = os.path.basename(filename)
            target_name = os.path.basename(mix_filename)
            # load wav
            anchor_wav, asr = torchaudio.load(filename)
            target_wav, tsr = torchaudio.load(mix_filename)
            assert asr == tsr, f'mixup sample rate should be the same {asr} vs. {tsr}'
            # get downbeat
            anchor_downbeat = np.load(os.path.join(self.tempo_folder, f'{anchor_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            target_downbeat = np.load(os.path.join(self.tempo_folder, f'{target_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            
            if len(anchor_downbeat) > 1 and len(target_downbeat) > 1:
                adp = int(anchor_downbeat[np.random.randint(0, len(anchor_downbeat) - 1)] * asr) 
                tdp = int(target_downbeat[np.random.randint(0, len(target_downbeat) - 1)] * tsr)
                anchor_wav = anchor_wav[..., adp:]
                target_wav = target_wav[..., tdp:]
                mix_len = min(anchor_wav.size(-1), target_wav.size(-1))
                if mix_len <= 100:
                    mix_wav, _ = torchaudio.load(filename)
                    anchor_wav = mix_wav[::]
                    target_wav = mix_wav[::]
                else:
                    anchor_wav = anchor_wav[..., :mix_len]
                    target_wav = target_wav[..., :mix_len]
                    p = np.random.beta(5,5)
                    mix_wav = p * anchor_wav + (1-p) * target_wav
            else:
                mix_wav = anchor_wav
                # normalize
            if self.return_all_wav:
                anchor_wav = self.normalize_wav(anchor_wav)
                target_wav = self.normalize_wav(target_wav)
                anchor_wav = anchor_wav[..., :self.segment_length]
                target_wav = target_wav[..., :self.segment_length]
                anchor_wav = torch.nn.functional.pad(anchor_wav, (0, self.segment_length - anchor_wav.size(1)), 'constant', 0.)
                target_wav = torch.nn.functional.pad(target_wav, (0, self.segment_length - target_wav.size(1)), 'constant', 0.)
                # get mel
                anchor_melspec, _, _ = self.STFT.mel_spectrogram(anchor_wav)
                anchor_melspec = anchor_melspec[0].T
                target_melspec, _, _ = self.STFT.mel_spectrogram(target_wav)
                target_melspec = target_melspec[0].T

                if anchor_melspec.size(0) < self.target_length:
                    anchor_melspec = torch.nn.functional.pad(anchor_melspec, (0,0,0,self.target_length - anchor_melspec.size(0)), 'constant', 0.)
                else:
                    anchor_melspec = anchor_melspec[0: self.target_length, :]
                
                if anchor_melspec.size(-1) % 2 != 0:
                    anchor_melspec = anchor_melspec[:, :-1]
                
                if target_melspec.size(0) < self.target_length:
                    target_melspec = torch.nn.functional.pad(target_melspec, (0,0,0,self.target_length - target_melspec.size(0)), 'constant', 0.)
                else:
                    target_melspec = target_melspec[0: self.target_length, :]

                if target_melspec.size(-1) % 2 != 0:
                    target_melspec = target_melspec[:, :-1]
                
                mix_wav, _ = torchaudio.load(filename) # unmix one for latent mixup

            y = self.normalize_wav(mix_wav)
            y = self.random_segment_wav(y)
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()


    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx] if len(self.label) > 0 else None
        # mixup
        if random.random() < self.mixup:
            wav_folder = os.path.dirname(f)
            anchor_name = os.path.basename(f)
            tempo_group = self.tempo_map['tempo'][self.tempo_map['map'][anchor_name]]
            if len(tempo_group) <= 1:
                mix_f = None
            else:
                mix_f = np.random.choice(tempo_group)
                mix_f = os.path.join(wav_folder, mix_f)
        else:
            mix_f = None
        # get data
        if self.return_all_wav:
            waveform, fbank,fbank_1, fbank_2 = self.get_mel(f, mix_f)
            data_dict['fbank_1'] = fbank_1
            data_dict['fbank_2'] = fbank_2
        else:
            waveform, fbank = self.get_mel(f, mix_f)
        if lf is not None:
            with open(lf, 'r') as lff:
                label_data = json.load(lff)
                text = label_data['text'][0]
        else:
            text = ""
        # construct dict
        data_dict['fname'] = os.path.basename(f).split('.')[0]
        data_dict['fbank'] = fbank
        data_dict['waveform'] = waveform
        data_dict['text'] = text
        return data_dict

    def __len__(self):
        return self.total_len



class DS_10283_2325_Dataset(AudiostockDataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False)  

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        # Open the file and read lines
        with open(file_path, "r") as fp:
            data_json = json.load(fp)
            data = data_json["data"]

        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        # dataset_name = "train" #filename.split('-')[1]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, "wav_files"))

        for entry in data:

            # get audio data path
            prompt_file_path = os.path.join(wav_directory, entry["audio_prompt"])
            response_file_path = os.path.join(wav_directory, entry["audio_response"])
            entry['wav'] = prompt_file_path
            entry["response"] = response_file_path

        # self.data = data
        self.label = data
        return data

    def read_wav(self, filename, frame_offset):

        audio_data, sr =torchaudio.load(filename, frame_offset =  frame_offset*48000, num_frames = 480000)

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(audio_data, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y
    
    def get_mel(self, filename, mix_filename = None, frame_offset = 0):
        # mixup
        y = self.read_wav(filename, frame_offset)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()
            
    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx]
        if lf is not None:
            text = lf['text']
        else:
            text = ""        
        
        
        prompt, fbank_prompt = self.get_mel(f["wav"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        # construct dict
        data_dict['fname'] = os.path.basename(lf['text']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict



class Audiostock_splited_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False)  

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        # Open the file and read lines
        data = []
        with open(file_path, "r") as fp:
            data_json = json.load(fp)

        for key, inner_dict in data_json.items():
            new_dict = inner_dict.copy()  # Create a copy of the inner dictionary
            new_dict['id'] = key  # Add the key from the outer dictionary
            data.append(new_dict)        

        prompt = self.config["path"]["prompt"]
        response = self.config["path"]["response"]

        # here we need logic taht only leaves in data entries that have nothe prompr and reposonse


        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        dataset_name = filename.split('_')[0]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, dataset_name+"_splited_16khz"))

        # Filter out entries where prompt or response is 0
        data = [entry for entry in data if entry[prompt+".wav"] != 0 and entry[response+".wav"] != 0]

        # iterate to get wav directories and append lable info
        for entry in data:
            prompt_file_path = os.path.join(wav_directory, entry["id"], prompt+".wav")

            response_file_path = os.path.join(wav_directory, entry["id"], response+".wav")
            entry['prompt'] = prompt_file_path
            entry["response"] = response_file_path

            if label_path is not None:
                lp = os.path.join(label_path, entry["id"] + '.json')
                assert os.path.exists(lp), f'the label file {lp} does not exists.'
                with open(lp, "r") as fp:
                    label_json = json.load(fp)
                entry.update(label_json)   

        new_data = []
        entries_to_remove = []  # List to store entries to be removed
        for entry in data:
            entry['frame_offset'] = 0
            # cut long files and take 10 seconds. first 30 seconds only if available
            duration = entry['original_data']["audio_size"]
            if duration > 10:  
                num_copies = int((min(duration,600)-10) / 10)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i+1) * 10
                    new_data.append(new_entry)

            # find very short files
            if duration < 0.2:  
                entries_to_remove.append(entry)
        
        # Remove the entries from data
        for entry in entries_to_remove:
            data.remove(entry)

        # add new entries
        data.extend(new_data)

        return data


    def read_wav(self, filename, frame_offset):

        y, _ =torchaudio.load(filename, frame_offset =  frame_offset*16000, num_frames = 160000)

        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        # lf = self.label[idx]
        # if lf is not None:
        #     text = lf['text']
        # else:
        #     text = ""        
        text = f['text']
        
        prompt, fbank_prompt = self.get_mel(f["prompt"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        # construct dict
        data_dict['fname'] = os.path.basename(f['id']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict


class Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False)  

    def read_datafile(self, dataset_path, label_path, train):

        data = []

        # Iterate over entries in the dataset
        for entry in os.listdir(dataset_path):
            entry_path = os.path.join(dataset_path, entry)
            
            # Check if metadata.yaml file exists
            lp = os.path.join(entry_path, 'metadata_updated.yaml')
            if os.path.exists(lp):
                pass
            else:
                continue

            # Read and load the YAML file
            with open(lp, "r") as fp:
                label_yaml = yaml.safe_load(fp)
            
            # Append the loaded data to the list
            data.append(label_yaml)
        
        filtered_data = []

        prompt = self.config["path"]["prompt"]
        response = self.config["path"]["response"]

        # Create pairs of prompts and responses
        for entry in data:

            prompts = []
            responses = []

            wav_directory = os.path.join(dataset_path, entry['audio_dir'])

            # Collect all prompts and responses
            for name, stem in entry['stems'].items():
                file_path = os.path.join(wav_directory, name + ".flac")
                
                if os.path.exists(file_path):
                    if stem['inst_class'] == prompt:
                        prompts.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})

                    elif stem['inst_class'] == response:
                        responses.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})
                else:
                    continue
               
            # Pair each prompt with each response
            for prompt_entry in prompts:
                for response_entry in responses:
                    
                    # Compare active segments
                    prompt_segments = set(prompt_entry['active_segments'])
                    response_segments = set(response_entry['active_segments'])
                    shared_segments = sorted(prompt_segments.intersection(response_segments))
                    
                    # Create a new entry for each shared segment
                    if shared_segments:
                        for segment in shared_segments:
                            new_entry = entry.copy()
                            
                            new_entry['prompt'] = prompt_entry['path']
                            new_entry['response'] = response_entry['path']
                            new_entry['frame_offset'] = segment
                            filtered_data.append(new_entry)
                    else:
                        pass
                        # print("No shared active segments. Skipping entry.")

        return filtered_data


    def read_wav(self, filename, frame_offset):

        y, sr =torchaudio.load(filename, frame_offset =  frame_offset*44100, num_frames = 441000)

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)

        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        
        prompt, fbank_prompt = self.get_mel(f["prompt"], None, int(f["frame_offset"]))
        response, fbank_response = self.get_mel(f["response"], None, int(f["frame_offset"]))

        # construct dict
        data_dict['fname'] = f['audio_dir'].split('/')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        # data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict


class MultiSource_Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False)  

    def get_duration_sec(self, file, cache=False):

        if not os.path.exists(file):
            # File doesn't exist, return 0
            return 0
        try:
            # Attempt to read cached duration from a file
            with open(file + ".dur", "r") as f:
                duration = float(f.readline().strip("\n"))
        except FileNotFoundError:
            # If cached duration is not found, use torchaudio to find the actual duration
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            if cache:
                # Cache the duration for future use
                with open(file + ".dur", "w") as f:
                    f.write(str(duration) + "\n")
        return duration
    
    def filter(self, tracks, audio_files_dir):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            # files = librosa.util.find_files(f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])
            # Manually create the list of file paths based on stems defined in the configuration
            files = [os.path.join(track_dir, stem + ".wav") for stem in self.config["path"]["stems"]]
            
            
            # skip if there are no sources per track
            if not files:
                continue
            
            durations_track = np.array([self.get_duration_sec(file, cache=True) * self.config['preprocessing']['audio']['sampling_rate'] for file in files]) # Could be approximate
            
            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] < 10).any():
                continue
            
            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] >= 600).any():
                print("skiping_file:", track)
                continue
            
            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                print(f"{track} skipped because sources are not aligned!")
                print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])
        
        print(f"sr={self.config['preprocessing']['audio']['sampling_rate']}, min: {10}, max: {600}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")

        return keep, durations, np.cumsum(np.array(durations))

    def read_datafile(self, dataset_path, label_path, train):
        data = []
        # Load list of tracks and starts/durations
        tracks = os.listdir(dataset_path)
        print(f"Found {len(tracks)} tracks.")
        keep, durations, cumsum = self.filter(tracks, dataset_path)

        # Assuming keep, durations, and cumsum are lists of the same length
        for idx in range(len(keep)):
            # Construct a dictionary for each track with its name, duration, and cumulative sum
            track_info = {
                'wav_path': os.path.join(dataset_path,keep[idx]),
                'duration': durations[idx],
                # 'cumsum': cumsum[idx]
            }
            # Append the dictionary to the data list
            data.append(track_info)

        return data


    def read_wav(self, filename, frame_offset):

        y, sr =torchaudio.load(filename, frame_offset =  int (frame_offset*22050), num_frames = int(22050*10.24)) # taking a little longer sample than 10 sec 

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)

        # normalize
        # y = self.normalize_wav(y) ##### don't do this because mix matters!!!!
        # segment
        # y = self.random_segment_wav(y)
            
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_index_offset(self, item):

        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.segment_length // 2

        
        start, end = 0.0, self.data[item]["duration"]  # start and end of current song
        
        offset = np.random.randint(start, end)  # random shif

        if offset > end - self.segment_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        # assert (
        #         start <= offset <= end - self.segment_length
        # ), f"Offset {offset} not in [{start}, {end - self.segment_length}]. End: {end}, SL: {self.segment_length}, Index: {item}"
        
        offset = offset/self.config['preprocessing']['audio']['sampling_rate']
        return item, offset

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]

        index, frame_offset = self.get_index_offset(idx)
        # wav = self.get_song_chunk(index, offset)
        # return self.transform(torch.from_numpy(wav))

        audio_list = []
        fbank_list = []
        for stem in self.config["path"]["stems"]:

            audio, fbank = self.get_mel(os.path.join(f["wav_path"],stem+".wav"), None, frame_offset)
            audio_list.append(audio[np.newaxis, :])  # Expand dims for audio
            fbank_list.append(fbank[np.newaxis, :])  # Expand dims for fbank

        
        # construct dict
        data_dict['fname'] = f['wav_path'].split('/')[-1]+"_from_"+str(int(frame_offset))
        data_dict['fbank'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform'] = np.concatenate(audio_list, axis=0)
        # data_dict['text'] = text

        # Mix audio and fbank features by summing; ensure same length and proper alignment
        # :TODO careful with potential clipping
        data_dict['mix'] = np.sum(data_dict['waveform'], axis=0)
        data_dict['fbank_mix'] = np.sum(data_dict['fbank'], axis=0)


        return data_dict


class Dataset(Dataset):
    def __init__(
        self,
        preprocess_config=None,
        train_config=None,
        samples_weight=None,
        train=True,
        shuffle=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        self.datapath = (
            preprocess_config["path"]["train_data"]
            if (train)
            else preprocess_config["path"]["test_data"]
        )

        self.data = []
        if type(self.datapath) is str:
            with open(self.datapath, "r") as fp:
                data_json = json.load(fp)
            self.data = data_json["data"]
        elif type(self.datapath) is list:
            for datapath in self.datapath:
                with open(datapath, "r") as fp:
                    data_json = json.load(fp)
                self.data += data_json["data"]
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.samples_weight = samples_weight

        # if(self.samples_weight is not None):
        #     print("+Use balance sampling on the mixup audio")
        #     self.sample_weight_index = list(range(len(self.samples_weight)))
        #     self.samples_weight /= np.sum(self.samples_weight)

        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]

        # try:
        #     self.rolling = train_config["augmentation"]["rolling"]
        #     if(self.rolling):
        #         print("+ Using rolling augmentation")
        # except:
        #     self.rolling = False

        # if(self.mixup > 0 and self.samples_weight is not None):
        #     self.mix_sample_idx_queue = np.random.choice(self.sample_weight_index, p=self.samples_weight, size=1000)

        # No augmentation during evaluation
        if train == False:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.sampling_rate = preprocess_config["preprocessing"]["audio"][
            "sampling_rate"
        ]
        # self.segment_label_path = preprocess_config["path"]["segment_label_path"]
        # self.clip_label_path = preprocess_config["path"]["clip_label_path"]
        self.hopsize = self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.target_length = self.preprocess_config["preprocessing"]["mel"][
            "target_length"
        ]
        self.use_blur = self.preprocess_config["preprocessing"]["mel"]["blur"]

        # try: self.label_norm = self.preprocess_config["preprocessing"]["label"]["norm"]
        # except: self.label_norm=False
        # try: self.label_threshold = self.preprocess_config["preprocessing"]["label"]["threshold"]
        # except: self.label_threshold=False

        self.segment_length = int(self.target_length * self.hopsize)

        try:
            self.segment_size = self.preprocess_config["preprocessing"]["audio"][
                "segment_size"
            ]
            self.target_length = int(self.segment_size / self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None

        # try:
        #     self.label_use_original_ground_truth = self.preprocess_config["preprocessing"]["label"]["label_use_original_ground_truth"]
        #     if(self.label_use_original_ground_truth): print("==> Use ground truth label: %s" % self.label_use_original_ground_truth)
        # except:
        #     print("Use machine labels")
        #     self.label_use_original_ground_truth=False

        # try:
        #     self.label_use_both_original_gt_and_machine_labels = self.preprocess_config["preprocessing"]["label"]["label_use_both_original_gt_and_machine_labels"]
        #     if(self.label_use_both_original_gt_and_machine_labels): print("==> Use both ground truth label and machine labels at the same time: %s" % self.label_use_both_original_gt_and_machine_labels)
        # except:
        #     self.label_use_both_original_gt_and_machine_labels=False

        print(
            "Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s"
            % (self.mixup, self.timem, self.freqm, self.use_blur)
        )

        # dataset spectrogram mean and std, used to normalize the input
        # self.norm_mean = preprocess_config["preprocessing"]["mel"]["mean"]
        # self.norm_std = preprocess_config["preprocessing"]["mel"]["std"]

        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = False
        self.noise = False
        if self.noise == True:
            print("now use noise augmentation")

        self.index_dict = make_index_dict(
            preprocess_config["path"]["class_label_index"]
        )
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

        # self.class_reweight_matrix = np.load(preprocess_config["path"]["class_reweight_arr_path"])

        self.id2label, self.id2num, self.num2label = self.build_id_to_label()

    def build_id_to_label(self):
        ret = {}
        id2num = {}
        num2label = {}
        df = pd.read_csv(self.preprocess_config["path"]["class_label_index"])
        for _, row in df.iterrows():
            index, mid, display_name = row["index"], row["mid"], row["display_name"]
            ret[mid] = display_name
            id2num[mid] = index
            num2label[index] = display_name
        return ret, id2num, num2label

    def resample(self, waveform, sr):
        if sr == 16000:
            return waveform
        if sr == 32000 and self.sampling_rate == 16000:
            waveform = waveform[::2]
            return waveform
        if sr == 48000 and self.sampling_rate == 16000:
            waveform = waveform[::3]
            return waveform
        else:
            raise ValueError(
                "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
                % (sr, self.sampling_rate)
            )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5

    def random_segment_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - self.segment_length) <= 0:
            return waveform

        random_start = int(
            self.random_uniform(0, waveform_length - self.segment_length)
        )
        return waveform[:, random_start : random_start + self.segment_length]

    def pad_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == self.segment_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, self.segment_length))
        rand_start = int(self.random_uniform(0, self.segment_length - waveform_length))
        # rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower

        waveform, sr = torchaudio.load(filename)  # Faster!!!
        waveform = waveform.numpy()[0, ...]

        waveform = self.resample(waveform, sr)
        waveform = self.normalize_wav(waveform)
        waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]

        waveform = self.random_segment_wav(waveform)
        waveform = self.pad_wav(waveform)

        return waveform

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform = self.read_wav_file(filename)
        # mixup
        else:
            waveform1 = self.read_wav_file(filename)
            waveform2 = self.read_wav_file(filename2)

            # if waveform1.shape[1] != waveform2.shape[1]:
            #     if waveform1.shape[1] > waveform2.shape[1]:
            #         # padding
            #         temp_wav = np.zeros((1, waveform1.shape[1]))
            #         temp_wav[0, 0:waveform2.shape[1]] = waveform2
            #         waveform2 = temp_wav
            #     else:
            #         # cutting
            #         waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(5, 5)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = self.normalize_wav(mix_waveform)

        # if self.segment_length > waveform.shape[1]:
        #     # padding
        #     temp_wav = np.zeros((1, self.segment_length))
        #     temp_wav[:, :waveform.shape[1]] = waveform
        #     waveform = temp_wav
        # else:
        #     # cutting
        #     waveform = waveform[:, :self.segment_length]

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        fbank, log_magnitudes_stft, energy = Audio.tools.get_mel_from_wav(
            waveform, self.STFT
        )

        fbank = torch.FloatTensor(fbank.T)
        log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

        fbank, log_magnitudes_stft = self._pad_spec(fbank), self._pad_spec(
            log_magnitudes_stft
        )

        if filename2 == None:
            return fbank, log_magnitudes_stft, 0, waveform
        else:
            return fbank, log_magnitudes_stft, mix_lambda, waveform

    def _pad_spec(self, fbank):
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        if fbank.size(-1) % 2 != 0:
            fbank = fbank[..., :-1]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # if(self.label_use_both_original_gt_and_machine_labels):
        #     if(self.make_decision(0.5)):
        #         self.label_use_original_ground_truth = True
        #     else:
        #         self.label_use_original_ground_truth = False

        (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        ) = self.feature_extraction(index)

        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += self.label_indices_to_text(mix_datum, label_indices)

        t_step = fbank.size(0)
        waveform = waveform[..., : int(self.hopsize * t_step)]

        # if(self.rolling and self.make_decision(1.0)):
        #     rand_roll = int(self.random_uniform(0, t_step))
        #     fbank = torch.roll(fbank, shifts=rand_roll, dims=0)
        #     log_magnitudes_stft = torch.roll(log_magnitudes_stft, shifts=rand_roll, dims=0)
        #     waveform = torch.roll(waveform, shifts = rand_roll * self.hopsize, dims=-1)

        # fbank = self.aug(fbank)

        # Reconsider whether or not need this step?
        # if(not self.label_use_original_ground_truth):
        # seg_label = self.process_labels(seg_label)
        # else:

        # if(self.label_use_original_ground_truth):
        #     if(len(label_indices.shape) <= 1):
        #         seg_label = label_indices[None,...]
        #     seg_label = np.repeat(seg_label.numpy(), 1056, 0)
        #     seg_label = seg_label[:self.target_length,:]
        #     clip_label = label_indices

        return (
            fbank.float(),
            log_magnitudes_stft.float(),
            label_indices.float(),
            fname,
            waveform.float(),
            text,
        )  # clip_label.float()

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def label_indices_to_text(self, datum, label_indices):
        if "caption" in datum.keys():
            return datum["caption"]
        name_indices = torch.where(label_indices > 0.1)[0]
        # description_header = "This audio contains the sound of "
        description_header = ""
        labels = ""
        for id, each in enumerate(name_indices):
            if id == len(name_indices) - 1:
                labels += "%s." % self.num2label[int(each)]
            else:
                labels += "%s, " % self.num2label[int(each)]
        return description_header + labels

    # def process_labels(self, seg_label):
    #     # Unify the scores in the label
    #     if(self.label_norm):
    #         seg_label = seg_label / self.class_reweight_matrix[None,...]
    #     # Remove the noise in the label
    #     seg_label[seg_label < self.label_threshold] = 0.0
    #     return seg_label

    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                if random.random() < self.mixup:
                    datum = self.data[index]
                    ###########################################################
                    # if(self.samples_weight is None):
                    mix_sample_idx = random.randint(0, len(self.data) - 1)
                    # else:
                    #     if(self.mix_sample_idx_queue.shape[0] < 10):
                    #         self.mix_sample_idx_queue = np.random.choice(self.sample_weight_index, p=self.samples_weight, size=1000)
                    #     mix_sample_idx = self.mix_sample_idx_queue[-1]
                    #     self.mix_sample_idx_queue = self.mix_sample_idx_queue[:-1]
                    mix_datum = self.data[mix_sample_idx]
                    ###########################################################
                    # get the mixed fbank
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"], mix_datum["wav"]
                    )
                    # initialize the label
                    label_indices = np.zeros(self.label_num)
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += mix_lambda
                    for label_str in mix_datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += (
                            1.0 - mix_lambda
                        )

                    # if(not self.label_use_original_ground_truth):
                    #     # [53, 527], [1, 527]
                    #     try:
                    #         clip_label1 = self.read_machine_label(index)
                    #         clip_label2 = self.read_machine_label(mix_sample_idx)

                    #         # seg_label = mix_lambda * seg_label1 + (1-mix_lambda) * seg_label2
                    #         clip_label = mix_lambda * clip_label1 + (1-mix_lambda) * clip_label2
                    #     except Exception as e:
                    #         print(e)
                    #         clip_label = label_indices[None,...]
                    # else:
                    #     # seg_label = np.repeat(label_indices[None,...], 53, axis=0)
                    #     clip_label = label_indices[None,...]
                else:
                    datum = self.data[index]
                    label_indices = np.zeros(self.label_num)
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"]
                    )
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                    # if(not self.label_use_original_ground_truth):
                    #     try:
                    #         clip_label = self.read_machine_label(index)
                    #     except Exception as e:
                    #         print(e)
                    #         clip_label = label_indices[None,...]
                    # else:
                    #     # seg_label = np.repeat(label_indices[None,...], 53, axis=0)
                    #     clip_label = label_indices[None,...]
                    mix_datum = None
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                print("feature_extraction", e)
                continue

        # The filename of the wav file
        fname = datum["wav"]

        # seg_label = torch.FloatTensor(seg_label)
        # clip_label = torch.FloatTensor(clip_label)
        clip_label = None

        return (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        )

    # def read_machine_label(self, index):
    #     # Read the clip-level or segment-level labels
    #     while(True):
    #         try:
    #             clip_label = self.read_label(index)
    #             return clip_label
    #         except Exception as e:
    #             print("read_machine_label", e)
    #             if(index == len(self.data)-1): index = 0
    #             else: index += 1

    def aug(self, fbank):
        assert torch.min(fbank) < 0
        fbank = fbank.exp()
        ############################### Blur and Spec Aug ####################################################
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        # self.use_blur = False
        if self.use_blur:
            fbank = self.blur(fbank)
        if self.freqm != 0:
            fbank = self.frequency_masking(fbank, self.freqm)
        if self.timem != 0:
            fbank = self.time_masking(fbank, self.timem)  # self.timem=0
        #############################################################################################
        fbank = (fbank + 1e-7).log()
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return fbank

    # def read_label(self, index):
    #     if("clip_label" in self.data[index].keys()):
    #         clip_label_fname = self.data[index]["clip_label"]
    #     else:
    #         wav_name = self.data[index]['wav']
    #         label_fname = os.path.basename(wav_name).replace(".wav",".npy")
    #         clip_label_fname = os.path.join(self.clip_label_path, label_fname)

    #     if(not os.path.exists(clip_label_fname)):
    #         return None

    #     clip_label = np.load(clip_label_fname)

    #     # For the clip level label, add one more dimension
    #     if(len(clip_label.shape) <= 1):
    #         clip_label = clip_label[None,...]

    #     clip_label = self.process_labels(clip_label)
    #     # seg_label = self.process_labels(seg_label)

    #     return clip_label

    def __len__(self):
        return len(self.data)

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def blur(self, fbank):
        assert torch.min(fbank) >= 0
        kernel_size = int(self.random_uniform(1, self.melbins))
        fbank = torchvision.transforms.functional.gaussian_blur(
            fbank, kernel_size=[kernel_size, kernel_size]
        )
        return fbank

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        fbank[:, mask_start : mask_start + mask_len, :] *= 0.0
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        fbank[:, :, mask_start : mask_start + mask_len] *= 0.0
        return fbank


def balance_test():
    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_unbal_train_data.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    # dataset = AudioDataset(samples_weight = None, train=True)
    dataset = AudioDataset(samples_weight=samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save(
        "balanced_with_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )
    # np.save("balanced_with_no_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    ######################################
    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save(
        "balanced_with_no_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )

    ######################################

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=8,
        # sampler=sampler
    )

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save("no_balance.npy", label_indices_total.cpu().detach().numpy() / 2000)


def check_batch(batch):
    import soundfile as sf
    import matplotlib.pyplot as plt

    save_path = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/output/temp"
    os.makedirs(save_path, exist_ok=True)
    fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
    for fb, wv, description in zip(fbank, waveform, text):
        sf.write(
            save_path + "/" + "%s.wav" % description.replace(" ", "_")[:30], wv, 16000
        )
        plt.imshow(np.flipud(fb.cpu().detach().numpy().T), aspect="auto")
        plt.savefig(save_path + "/" + "%s.png" % description.replace(" ", "_")[:30])


if __name__ == "__main__":

    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    preprocess_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/preprocess.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )
    train_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/train.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/datafiles/audiocaps_train_label.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(
        samples_weight=samples_weight,
        train=True,
        train_config=train_config,
        preprocess_config=preprocess_config,
    )

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        check_batch(each)
        break

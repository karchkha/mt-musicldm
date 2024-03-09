import ast
import json
import logging
import math
import os
import random
import h5py
from dataclasses import dataclass
import braceexpand
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path
import wget
import tempfile
import copy
from contextlib import suppress

from clap_module.utils import get_tar_path_from_dataset_name, dataset_split
from clap_module.utils import load_p, load_class_label
from clap_module import tokenize as clip_tokenizer
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import BartTokenizer

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenizer(text, tmodel="roberta", max_length=77):
    """tokenizer for different models
    tmodel is default to roberta as it is the best model for our task
    max_length is default to 77 from the OpenAI CLIP parameters
    We assume text to be a single string, but it can also be a list of strings
    """
    if tmodel == "transformer":
        return clip_tokenizer(text).squeeze(0)

    elif tmodel == "bert":
        result = bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

    elif tmodel == "roberta":
        result = roberta_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

    elif tmodel == "bart":
        result = bart_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}


# initizlied the audioset map
_AUDIOSET_MAP_PATH = os.path.join(Path(__file__).parent, "audioset_textmap.npy")
_AUDIOSET_MAP = np.load(_AUDIOSET_MAP_PATH, allow_pickle=True)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


# For Toy Dataset
class ToyDataset(Dataset):
    def __init__(self, index_path, ipc, config, eval_mode=False):
        """Toy Dataset for testing the audioset input with text labels
        Parameters
        ----------
            index_path: str
                the link to the h5 file of each audio
            idc: str
                the link to the npy file, the number of samples in each class
            config: dict
                the audio cfg file
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.audio_cfg = config["audio_cfg"]
        self.text_cfg = config["text_cfg"]
        self.fp = h5py.File(index_path, "r")
        self.ipc = np.load(ipc, allow_pickle=True)
        self.total_size = len(self.fp["audio_name"])
        self.classes_num = self.audio_cfg["class_num"]
        self.eval_mode = eval_mode

        if not eval_mode:
            self.generate_queue()
        else:
            self.queue = []
            for i in range(self.total_size):
                target = self.fp["target"][i]
                if np.sum(target) > 0:
                    self.queue.append(i)
            self.total_size = len(self.queue)
        logging.info("total dataset size: %d" % (self.total_size))
        logging.info("class num: %d" % (self.classes_num))

    def time_shifting(self, x):
        frame_num = len(x)
        shift_len = random.randint(0, frame_num - 1)
        new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis=0)
        return new_sample

    def generate_queue(self):
        self.queue = []
        while len(self.queue) < self.total_size:
            class_set = [*range(self.classes_num)]
            random.shuffle(class_set)
            self.queue += [
                self.ipc[d][random.randint(0, len(self.ipc[d]) - 1)] for d in class_set
            ]
        self.queue = self.queue[: self.total_size]

        logging.info("queue regenerated:%s" % (self.queue[-5:]))

    def crop_wav(self, x):
        crop_size = self.audio_cfg["crop_size"]
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos: crop_pos + crop_size]

    def prompt_text(self, target):
        events = _AUDIOSET_MAP[np.where(target > 0)]
        event_text = "The sounds of " + ", ".join(events[:-1]) + " and " + events[-1]
        text = tokenizer(event_text)[0]
        return text

    def __getitem__(self, index):
        """Load waveform, text, and target of an audio clip

        Parameters
        ----------
            index: int
                the index number
        Return
        ------
            output: dict {
                "hdf5_path": str,
                "index_in_hdf5": int,
                "audio_name": str,
                "waveform": list (audio_length,),
                "target": list (class_num, ),
                "text": torch.tensor (context_length,)
            }
                the output dictionary
        """
        s_index = self.queue[index]

        audio_name = self.fp["audio_name"][s_index].decode()
        # Hardcode here CHANGE
        hdf5_path = (
            self.fp["hdf5_path"][s_index]
            .decode()
            .replace(
                "../workspace",
                "/home/la/kechen/Research/ke_zsasp/workspace",
            )
        )
        r_idx = self.fp["index_in_hdf5"][s_index]
        target = self.fp["target"][s_index].astype(np.float32)
        text = self.prompt_text(target)
        with h5py.File(hdf5_path, "r") as f:
            waveform = int16_to_float32(f["waveform"][r_idx])[
                       : self.audio_cfg["clip_samples"]
                       ]
        assert (
                len(waveform) == self.audio_cfg["clip_samples"]
        ), "The sample length is not match"
        # Time shift
        # if (self.config.enable_time_shift) and (not self.eval_mode):
        #     waveform = self.time_shifting(waveform)
        # # Label Enhance
        # if (self.config.crop_size is not None) and (not self.eval_mode):
        #     waveform = self.crop_wav(waveform)
        # # the label enhance rate is fixed 0.5
        # if (self.config.enable_label_enhance) and (not self.eval_mode) and random.random() < 0.5:
        #     kidx = np.where(target)[0]
        #     for k in kidx:
        #         for add_key in self.class_map[k][1]:
        #             target[add_key] = 1.0
        #         if len(self.class_map[k][2]) > 0:
        #             add_key = random.choice(self.class_map[k][2])
        #             target[add_key] = 1.0

        # missing the text input
        mel_spec = get_mel(torch.from_numpy(waveform), self.audio_cfg)[None, :, :]
        mel_spec = torch.cat([mel_spec, mel_spec.clone(), mel_spec.clone(), mel_spec.clone()], dim=0).cpu().numpy()
        longer = random.choice([True, False])
        if longer == False:
            mel_spec[1:, :, :] = 0.0
        data_dict = {
            "hdf5_path": hdf5_path,
            "index_in_hdf5": r_idx,
            "audio_name": audio_name,
            "waveform": waveform,
            "class_label": target,
            "text": text,
            "longer": longer,
            "mel_fusion": mel_spec
        }
        return data_dict

    def __len__(self):
        return self.total_size


class Audiostock_10k_16khz_dataset(Dataset):
    def __init__(self, input_filename, transforms, args = None):
        logging.debug(f"Loading csv data from {input_filename}.")

        self.args = args

        df = []
        for datapath in input_filename:
            data =  self.read_datafile(datapath) 
            for dictionary in data:
                dictionary['__url__'] = datapath[:-5] + "/0"  # Removes the ".json" extension
            df += data


        # df = pd.read_csv(input_filename, sep=args.csv_separator)

        self.images = [item['wav'] for item in df]
        self.captions = [item['caption'] for item in df]
        self.transforms = transforms
        self.labels = [item['labels'] for item in df]
        self.frame_offset = [item['frame_offset'] for item in df]
        self.url = [item['__url__'] for item in df]
        logging.debug("Done loading data.")


        if args.class_index_dict is None:  ### putting empty dict if notheing was given
            args.class_index_dict  = {} #pd.DataFrame(columns=["index", "display_name"])

        self.class_index_dict=copy.deepcopy(args.class_index_dict)


    def read_datafile(self, file_path): # split):
        """
        Reads a JSON file and performs operations on its data.
        
        Args:
            file_path (str): The path to the JSON file.
            split (str): The split name to wrap the final dictionary.
            
        Returns:
            List[Dict[str, Any]]: The modified data as a list of dictionaries.
        """


        # Initialize an empty list to store dictionaries
        data = []

        # Open the file and read lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Iterate through each line
        for line in lines:
            # Create a dictionary with the desired key and value
            dictionary_entry = {'audio_file_name_on_website': line.strip()}  # Assuming you want to strip newline characters

            # Append the dictionary to the list
            data.append(dictionary_entry)


        dataset_directory = os.path.dirname(os.path.dirname(file_path))
        filename = os.path.basename(file_path)
        dataset_name = filename.split('-')[1]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, dataset_name))
        label_path = os.path.abspath(os.path.join(dataset_directory,"label"))

        new_data = []
        entries_to_remove = []  # List to store entries to be removed




        for i, entry in enumerate(data):

            #this is teporary will be removed when we correct dataset
            if dataset_name == "train":
                i = i+1
            else:
                i = i + 9005
                # if i == 10005:
                #     i=10004

            lp = os.path.join(label_path, str(i) + '.json')
            assert os.path.exists(lp), f'the label file {lp} does not exists.'
            with open(lp, 'r') as lff:
                label_data = json.load(lff)

            # get flac data pathe
            flac_file_path = os.path.join(wav_directory, str(i) + '.wav')
            entry['wav'] = flac_file_path

            entry['frame_offset'] = 0

            entry["caption"] = label_data.get("text", 'not available')[0]
            # get data category if given

            labels = label_data.get('tag',  'not available')
            if labels == 'not available':
                entry['labels'] = []
            elif isinstance(labels, list):
                entry['labels'] = labels
            else:
                try:
                    label_list = eval(labels)
                    if isinstance(label_list, list):
                        entry['labels'] = label_list
                    else:
                        entry['labels'] = []
                except (NameError, SyntaxError):
                    entry['labels'] = []

            # cut long files and take 10 seconds. first 30 seconds only if available
            orig_data = label_data.get('original_data', 0)
            
            duration = orig_data.get('audio_size', 0)


            if duration > 10:  
                num_copies = min(int((duration-10) / 10), 2)
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

        # # Wrap the final dictionary with the split name
        # result = {split: data}

        return data
    

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        #             preprocess,
        #     audio_ext=audio_ext,
        #     text_ext=text_ext,
        #     max_len=max_len,
        #     audio_cfg=model_cfg["audio_cfg"],
        #     class_index_dict=copy.deepcopy(args.class_index_dict),
        #     data_filling=args.data_filling,
        #     data_truncating=args.data_truncating,
        #     text_augment_selection=args.text_augment_selection,
        # )

        sample = {}
        audio_data, orig_sr =torchaudio.load(self.images[idx], frame_offset =  self.frame_offset[idx], num_frames = 160000)

        audio_data = audio_data[0] # taking left channel only

        # new_sr = 16000
        # resample_transform = torchaudio.transforms.Resample(orig_sr, new_sr)
        # audio_data = resample_transform(audio_data)

        new_sr = 48000
        resample_transform = torchaudio.transforms.Resample(orig_sr, new_sr)
        audio_data = resample_transform(audio_data)

        sample = get_audio_features(
            sample, audio_data, max_len = new_sr*10, data_truncating = self.args.data_truncating, data_filling = self.args.data_filling, audio_cfg = self.transforms["audio_cfg"]
        )

        machine_codes = self.labels[idx]  #str(self.captions[idx])

        if self.class_index_dict is not None:
            # https://stackoverflow.com/questions/48004243/how-to-share-large-read-only-dictionary-list-across-processes-in-multiprocessing
            # https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array
            # key, val = class_index_dict
            # key = key[:].split('\n')
            # _dict = {k: v for k, v in zip(key, val)}
            sample["class_label"] = np.zeros(len(self.class_index_dict.keys()))

            for label_str in machine_codes:
                try:    
                    sample["class_label"][int(self.class_index_dict[label_str])] = 1
                except:
                    pass                

            sample["class_label"] = torch.tensor(sample["class_label"]).float()

        if len(machine_codes)>0:
            sample["machine_codes"] = machine_codes[0] ### only show 1 lable if there are more
        else:
            if machine_codes == []:
                sample["machine_codes"] = ""
            else:
                sample["machine_codes"] = machine_codes

        sample['__url__'] = self.url[idx]

        text = str(self.captions[idx])
            
        text_augment_selection=self.args.text_augment_selection

        # For selecting augmented text from dataset
        if text_augment_selection is None or text_augment_selection == "none":
            texts = text
        else:
            raise NotImplementedError(
                f"text_augment_selection {text_augment_selection} not implemented"
            )
        sample["full_text"] = texts
    
        if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
            texts = random.choice(texts)
        sample["raw_text"] = texts
        sample["text"] = tokenizer(texts)  # text shape: [num_token]

        sample["audio_name"] = self.images[idx].split("/")[-1]
        # sample["text_name"] = sample["__key__"].split("/")[-1] + "." + text_ext
        sample["audio_orig_sr"] = orig_sr
        return sample


class DS_10283_2325_dataset(Dataset):
    def __init__(self, input_filename, transforms, args = None):
        logging.debug(f"Loading csv data from {input_filename}.")

        self.args = args

        df = []
        for datapath in input_filename:
            data =  self.read_datafile(datapath) 
            for dictionary in data:
                dictionary['__url__'] = datapath[:-5] + "/0"  # Removes the ".json" extension
            df += data


        # df = pd.read_csv(input_filename, sep=args.csv_separator)

        self.images = [item['wav'] for item in df]
        self.images_response = [item['response'] for item in df]
        self.captions = [item['caption'] for item in df]
        self.transforms = transforms
        self.labels = [item['labels'] for item in df]
        self.frame_offset = [item['frame_offset'] for item in df]
        self.url = [item['__url__'] for item in df]
        logging.debug("Done loading data.")


        if args.class_index_dict is None:  ### putting empty dict if notheing was given
            args.class_index_dict  = {} #pd.DataFrame(columns=["index", "display_name"])

        self.class_index_dict=copy.deepcopy(args.class_index_dict)


    def read_datafile(self, file_path): # split):
        """
        Reads a JSON file and performs operations on its data.
        
        Args:
            file_path (str): The path to the JSON file.
            split (str): The split name to wrap the final dictionary.
            
        Returns:
            List[Dict[str, Any]]: The modified data as a list of dictionaries.
        """


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

        return data
    

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        #             preprocess,
        #     audio_ext=audio_ext,
        #     text_ext=text_ext,
        #     max_len=max_len,
        #     audio_cfg=model_cfg["audio_cfg"],
        #     class_index_dict=copy.deepcopy(args.class_index_dict),
        #     data_filling=args.data_filling,
        #     data_truncating=args.data_truncating,
        #     text_augment_selection=args.text_augment_selection,
        # )

        sample = {}
        audio_data, sr =torchaudio.load(self.images[idx], frame_offset =  self.frame_offset[idx]*48000, num_frames = 480000)
        audio_data_response, sr =torchaudio.load(self.images_response[idx], frame_offset =  self.frame_offset[idx]*48000, num_frames = 480000)

        audio_data = audio_data[0] # taking left channel only
        audio_data_response = audio_data_response[0] # taking left channel only

        sample = get_audio_features(
            sample, audio_data, max_len = sr*10, data_truncating = self.args.data_truncating, data_filling = self.args.data_filling, audio_cfg = self.transforms["audio_cfg"]
        )

        sample["response"] = audio_data_response

        machine_codes = self.labels[idx]  #str(self.captions[idx])

        if self.class_index_dict is not None:
            # https://stackoverflow.com/questions/48004243/how-to-share-large-read-only-dictionary-list-across-processes-in-multiprocessing
            # https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array
            # key, val = class_index_dict
            # key = key[:].split('\n')
            # _dict = {k: v for k, v in zip(key, val)}
            sample["class_label"] = np.zeros(len(self.class_index_dict.keys()))

            for label_str in machine_codes:
                try:    
                    sample["class_label"][int(self.class_index_dict[label_str])] = 1
                except:
                    pass                

            sample["class_label"] = torch.tensor(sample["class_label"]).float()

        if len(machine_codes)>0:
            sample["machine_codes"] = machine_codes[0] ### only show 1 lable if there are more
        else:
            if machine_codes == []:
                sample["machine_codes"] = ""
            else:
                sample["machine_codes"] = machine_codes

        sample['__url__'] = self.url[idx]

        text = str(self.captions[idx])
            
        text_augment_selection=self.args.text_augment_selection

        # For selecting augmented text from dataset
        if text_augment_selection is None or text_augment_selection == "none":
            texts = text
        else:
            raise NotImplementedError(
                f"text_augment_selection {text_augment_selection} not implemented"
            )
        sample["full_text"] = texts
    
        if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
            texts = random.choice(texts)
        sample["raw_text"] = texts
        sample["text"] = tokenizer(texts)  # text shape: [num_token]

        sample["audio_name"] = self.images[idx].split("/")[-1]
        # sample["text_name"] = sample["__key__"].split("/")[-1] + "." + text_ext
        sample["audio_orig_sr"] = sr
        return sample


class Audiostock_splited_dataset(DS_10283_2325_dataset):
    def __init__(self, input_filename, transforms, args = None) -> None:
        super().__init__(input_filename, transforms, args = args)

    def read_datafile(self, dataset_path):
        file_path = dataset_path
        # Open the file and read lines
        data = []
        with open(file_path, "r") as fp:
            data_json = json.load(fp)

        for key, inner_dict in data_json.items():
            new_dict = inner_dict.copy()  # Create a copy of the inner dictionary
            new_dict['id'] = key  # Add the key from the outer dictionary
            data.append(new_dict)        

        prompt = self.args.prompt
        response = self.args.response

        # here we need logic taht only leaves in data entries that have nothe prompr and reposonse


        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        dataset_name = filename.split('_')[0]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, dataset_name+"_splited_16khz"))

        # Filter out entries where prompt or response is 0
        data = [entry for entry in data if entry[prompt+".wav"] != 0 and entry[response+".wav"] != 0]

        # iterate to get wav directories and append lable info
        label_path =os.path.join( self.args.datasetpath,"label")
        for entry in data:
            prompt_file_path = os.path.join(wav_directory, entry["id"], prompt+".wav")

            response_file_path = os.path.join(wav_directory, entry["id"], response+".wav")
            entry['wav'] = prompt_file_path
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

            entry["caption"] = entry.get("text", 'not available')[0]
            # get data category if given

            labels = entry.get('tag',  'not available')
            if labels == 'not available':
                entry['labels'] = []
            elif isinstance(labels, list):
                entry['labels'] = labels
            else:
                try:
                    label_list = eval(labels)
                    if isinstance(label_list, list):
                        entry['labels'] = label_list
                    else:
                        entry['labels'] = []
                except (NameError, SyntaxError):
                    entry['labels'] = []


            entry['frame_offset'] = 0
            # cut long files and take 10 seconds. first 30 seconds only if available
            duration = entry['original_data']["audio_size"]
            if duration > 10:  
                num_copies = int((min(duration,600)-10) / 10)  # max 600 sec
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

    def __getitem__(self, idx):

        #             preprocess,
        #     audio_ext=audio_ext,
        #     text_ext=text_ext,
        #     max_len=max_len,
        #     audio_cfg=model_cfg["audio_cfg"],
        #     class_index_dict=copy.deepcopy(args.class_index_dict),
        #     data_filling=args.data_filling,
        #     data_truncating=args.data_truncating,
        #     text_augment_selection=args.text_augment_selection,
        # )

        sample = {}
        sample_response = {}
        audio_data, sr =torchaudio.load(self.images[idx], frame_offset =  self.frame_offset[idx]*16000, num_frames = 160000)
        audio_data_response, sr =torchaudio.load(self.images_response[idx], frame_offset =  self.frame_offset[idx]*16000, num_frames = 160000)

        audio_data = audio_data[0] # taking left channel only
        audio_data_response = audio_data_response[0] # taking left channel only

        new_sr = 48000
        resample_transform = torchaudio.transforms.Resample(16000, new_sr)
        audio_data = resample_transform(audio_data)
        audio_data_response = resample_transform(audio_data_response)

        sample = get_audio_features(
            sample, audio_data, max_len = new_sr*10, data_truncating = self.args.data_truncating, data_filling = self.args.data_filling, audio_cfg = self.transforms["audio_cfg"]
        )

        sample_response = get_audio_features(
            sample_response, audio_data_response, max_len = new_sr*10, data_truncating = self.args.data_truncating, data_filling = self.args.data_filling, audio_cfg = self.transforms["audio_cfg"]
        )
        sample["response"] = sample_response["waveform"]

        machine_codes = self.labels[idx]  #str(self.captions[idx])

        if self.class_index_dict is not None:
            # https://stackoverflow.com/questions/48004243/how-to-share-large-read-only-dictionary-list-across-processes-in-multiprocessing
            # https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array
            # key, val = class_index_dict
            # key = key[:].split('\n')
            # _dict = {k: v for k, v in zip(key, val)}
            sample["class_label"] = np.zeros(len(self.class_index_dict.keys()))

            for label_str in machine_codes:
                try:    
                    sample["class_label"][int(self.class_index_dict[label_str])] = 1
                except:
                    pass                

            sample["class_label"] = torch.tensor(sample["class_label"]).float()

        if len(machine_codes)>0:
            sample["machine_codes"] = machine_codes[0] ### only show 1 lable if there are more
        else:
            if machine_codes == []:
                sample["machine_codes"] = ""
            else:
                sample["machine_codes"] = machine_codes

        sample['__url__'] = self.url[idx]

        text = str(self.captions[idx])
            
        text_augment_selection=self.args.text_augment_selection

        # For selecting augmented text from dataset
        if text_augment_selection is None or text_augment_selection == "none":
            texts = text
        else:
            raise NotImplementedError(
                f"text_augment_selection {text_augment_selection} not implemented"
            )
        sample["full_text"] = texts
    
        if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
            texts = random.choice(texts)
        sample["raw_text"] = texts
        sample["text"] = tokenizer(texts)  # text shape: [num_token]

        sample["audio_name"] = self.images[idx].split("/")[-1]
        # sample["text_name"] = sample["__key__"].split("/")[-1] + "." + text_ext
        sample["audio_orig_sr"] = sr
        return sample


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_dataset_size(shards, sizefilepath_=None, is_local=True):
    if isinstance(shards, list):
        size_list = []
        for s in shards:
            size_list.append(
                get_dataset_size(s, sizefilepath_=sizefilepath_, is_local=is_local)[0]
            )
    else:
        if not is_local:
            for n in dataset_split.keys():
                if n in shards.split("/"):
                    break
            for s in dataset_split[n]:
                if s in shards.split("/"):
                    break
            sizefilepath_ = f"./json_files/{n}/{s}/sizes.json"
        shards_list = list(braceexpand.braceexpand(shards))
        dir_path = os.path.dirname(shards)
        if sizefilepath_ is not None:
            sizes = json.load(open(sizefilepath_, "r"))
            total_size = sum(
                [
                    int(sizes[os.path.basename(shard.replace(".tar -", ".tar"))])
                    for shard in shards_list
                ]
            )
        else:
            sizes_filename = os.path.join(dir_path, "sizes.json")
            len_filename = os.path.join(dir_path, "__len__")
            if os.path.exists(sizes_filename):
                sizes = json.load(open(sizes_filename, "r"))
                total_size = sum(
                    [int(sizes[os.path.basename(shard)]) for shard in shards_list]
                )
            elif os.path.exists(len_filename):
                # FIXME this used to be eval(open(...)) but that seemed rather unsafe
                total_size = ast.literal_eval(open(len_filename, "r").read())
            else:
                raise Exception(
                    "Cannot find sizes file for dataset. Please specify the path to the file."
                )
                # total_size = None  # num samples undefined
                # some common dataset sizes (at time of authors last download)
                # cc3m-train: 2905954
                # cc12m: 10968539
                # LAION-400m: 407332084
        num_shards = len(shards_list)
    if isinstance(shards, list):
        return sum(size_list), len(shards)
    else:
        return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def sample_prop(sizefile, inputs, proportion, is_local=True):
    """
    Sample a proportion of the data.
    """
    file_path_dict = {
        os.path.split(inputs[i])[1]: os.path.split(inputs[i])[0]
        for i in range(len(inputs))
    }
    sampled_filepath_dict = {}
    sampled_size_dict = {}
    if not is_local:
        if os.path.exists("sizes.json"):
            os.remove("sizes.json")
        wget.download(sizefile, "sizes.json")
        sizefile = "sizes.json"
    with open(sizefile, "r", encoding="UTF-8") as f:
        load_dict = json.load(f)
    L = int(len(file_path_dict) * proportion)
    subkeys = random.sample(file_path_dict.keys(), L)
    for k in subkeys:
        sampled_size_dict[k] = load_dict[k]
        sampled_filepath_dict[k] = file_path_dict[k]
    return (
        sum(sampled_size_dict.values()),
        L,
        [os.path.join(v, k) for k, v in sampled_filepath_dict.items()],
        sampled_size_dict,
    )


def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    )(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=64,
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    require_grad: whether to require gradient for audio data.
        This is useful when we want to apply gradient-based classifier-guidance.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample


def select_text(json_dict_raw, text_augment_selection):
    # For selecting augmented text from dataset
    if text_augment_selection is None or text_augment_selection == "none":
        texts = json_dict_raw["text"]
    elif text_augment_selection == "all":
        if "text_augment_all" in json_dict_raw.keys():
            texts = json_dict_raw["text_augment_all"]
        else:
            texts = json_dict_raw["text"]
    elif text_augment_selection == "augment_only":
        if "text_augment_all" in json_dict_raw.keys():
            if json_dict_raw["text_augment_t5"] is None:
                texts = json_dict_raw["text"]
            else:
                texts = json_dict_raw["text_augment_t5"]
        else:
            texts = json_dict_raw["text"]
    else:
        raise NotImplementedError(
            f"text_augment_selection {text_augment_selection} not implemented"
        )
    return texts


def preprocess_single(
        sample,
        audio_ext,
        text_ext,
        max_len,
        audio_cfg,
        tmodel,
        class_index_dict,
        data_filling,
        data_truncating,
        text_augment_selection,
):
    """
    Preprocess a single sample for wdsdataloader.
    """
    audio_data, orig_sr = sample[audio_ext]
    audio_data = int16_to_float32_torch(float32_to_int16_torch(audio_data[0]))

    sample = get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg)
    del sample[audio_ext]

    json_dict_raw = sample[text_ext]

    texts = select_text(json_dict_raw, text_augment_selection)
    sample["full_text"] = texts

    if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
        texts = random.choice(texts)
    sample["raw_text"] = texts
    sample["text"] = tokenizer(texts, tmodel=tmodel)  # text shape: [num_token]
    if class_index_dict is not None:
        # https://stackoverflow.com/questions/48004243/how-to-share-large-read-only-dictionary-list-across-processes-in-multiprocessing
        # https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array

        # in case the re-written version is wrong, here is the old version:
        # sample["class_label"] = np.zeros(len(class_index_dict.keys()))
        # for x in json_dict_raw["tag"]:
        #     sample["class_label"][class_index_dict[x]] = 1
        # sample["class_label"] = torch.tensor(sample["class_label"]).float()

        class_labels = np.zeros(len(class_index_dict))
        class_labels[np.in1d(list(class_index_dict.keys()), json_dict_raw["tag"])] = 1
        sample["class_label"] = torch.tensor(class_labels).float()

    del sample[text_ext]
    sample["audio_name"] = sample["__key__"].split("/")[-1] + "." + audio_ext
    sample["text_name"] = sample["__key__"].split("/")[-1] + "." + text_ext
    sample["audio_orig_sr"] = orig_sr
    return sample


def collate_fn_with_preprocess(batch,
                               audio_ext,
                               text_ext,
                               max_len,
                               audio_cfg,
                               args,
                               ):
    """
    Collate function for wdsdataloader.
    batch: a list of dict, each dict is a sample
    """

    class_index_dict = copy.deepcopy(args.class_index_dict)  # To avoid deadlock in multiprocessing
    data_filling = args.data_filling
    data_truncating = args.data_truncating
    text_augment_selection = args.text_augment_selection
    tmodel = args.tmodel

    # concatenate values in each dictionary. if it is a tensor, concatenate. if it is a list, extend.
    data_preprocessed = []

    for sample in batch:
        data_preprocessed.append(
            preprocess_single(sample, audio_ext, text_ext, max_len, audio_cfg, tmodel, class_index_dict, data_filling,
                              data_truncating, text_augment_selection))

    batch_dict = {}
    for k in data_preprocessed[0].keys():
        if isinstance(data_preprocessed[0][k], dict):  # dealwith bert tokenizer output
            batch_dict[k] = {}
            for kk in data_preprocessed[0][k].keys():
                tmp = []
                for i in range(len(data_preprocessed)):
                    tmp.append(data_preprocessed[i][k][kk])
                batch_dict[k][kk] = torch.vstack(tmp)
        elif isinstance(data_preprocessed[0][k], torch.Tensor):
            batch_dict[k] = torch.stack([sample[k] for sample in data_preprocessed])
        elif isinstance(data_preprocessed[0][k], np.ndarray):
            batch_dict[k] = torch.tensor(np.stack([sample[k] for sample in data_preprocessed]))
        else:
            batch_dict[k] = [sample[k] for sample in data_preprocessed]
    del data_preprocessed
    return batch_dict


def get_wds_dataset(
        args,
        model_cfg,
        is_train,
        audio_ext="flac",
        text_ext="json",
        max_len=480000,
        proportion=1.0,
        sizefilepath_=None,
        is_local=None,
):
    """
    Get a dataset for wdsdataloader.
    """
    if is_local is None and (not args.remotedata is None):
        is_local = not args.remotedata

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    if not sizefilepath_ is None:
        sizefilepath = sizefilepath_
    else:
        sizefilepath = os.path.join(os.path.dirname(input_shards[0]), "sizes.json")

    if proportion != 1.0:
        num_samples, num_shards, input_shards, _ = sample_prop(
            sizefilepath, input_shards, proportion, is_local=is_local
        )
    else:
        num_samples, num_shards = get_dataset_size(
            input_shards, sizefilepath_=sizefilepath_, is_local=is_local
        )

    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    "Currently, number of dataset samples must be specified for training dataset. "
                    "Please specify via `--train-num-samples` if no dataset length info present."
                )
        else:
            num_samples = (
                    args.val_num_samples or 0
            )  # eval will just exhaust the iterator if not specified

    pipeline = [wds.SimpleShardList(input_shards)]
    # at this point we have an iterator over all the shards
    # TODO: (yusong): add a if statement of distributed. If not, we don't need to split_by_node
    if is_train or args.parallel_eval:
        pipeline.extend(
            [
                wds.detshuffle(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                    rng=random.Random(args.seed),
                ),
                # wds.repeatedly,  # FIXME determine if this is beneficial
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )

    pipeline.append(
        wds.decode(wds.torch_audio),
    )

    pipeline.append(
        wds.batched(
            args.batch_size,
            partial=not (is_train or args.parallel_eval),
            collation_fn=partial(collate_fn_with_preprocess,
                                 audio_ext=audio_ext,
                                 text_ext=text_ext,
                                 max_len=max_len,
                                 audio_cfg=model_cfg['audio_cfg'],
                                 args=args,
                                 ),

        )
    )

    dataset = wds.DataPipeline(*pipeline)
    if is_train or args.parallel_eval:
        # (yusong): Currently parallel evaluation will be not precise as we are repeat the last few samples.
        # (yusong): See comments below.
        # roll over and repeat a few samples to get same number of full batches on each node
        global_batch_size = args.batch_size * args.world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = math.ceil(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    kwargs = {}
    if args.horovod:  # multi-node training on summit
        kwargs["multiprocessing_context"] = "forkserver"

    if is_train:
        if args.prefetch_factor:
            prefetch_factor = args.prefetch_factor
        else:
            prefetch_factor = max(2, args.batch_size // args.workers)
    else:
        prefetch_factor = 2

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        **kwargs
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)


def wds_batch_list2dict(
        batch,
        keys=[
            "__url__",
            "__key__",
            "waveform",
            "text",
            "raw_text",
            "audio_name",
            "text_name",
            "audio_orig_sr",
        ],
):
    """
    Return a dictionary of the batch, with keys as the names of the fields.
    """
    assert len(keys) == len(
        batch
    ), "batch must have same number of keys as keys argument"
    return {keys[i]: batch[i] for i in range(len(batch))}



def get_toy_dataset(args, model_cfg, is_train):
    index_path = args.train_data if is_train else args.val_data
    ipc_path = args.train_ipc if is_train else args.val_ipc
    assert index_path and ipc_path
    eval_mode = not is_train
    dataset = ToyDataset(index_path, ipc_path, model_cfg, eval_mode=eval_mode)

    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset, shuffle=False)
        if args.distributed and is_train
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_Audiostock_10k_16khz_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    dataset = Audiostock_10k_16khz_dataset(
        input_filename,
        preprocess_fn,
        args=args
    )    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_DS_10283_2325(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    dataset = DS_10283_2325_dataset(
        input_filename,
        preprocess_fn,
        args=args
    )    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_Audiostock_splited(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    dataset = Audiostock_splited_dataset(
        input_filename,
        preprocess_fn,
        args=args
    )    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_dataset_fn(dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "toy":
        return get_toy_dataset
    elif dataset_type == "Audiostock-10k-16khz":
        return get_Audiostock_10k_16khz_dataset
    elif dataset_type == 'DS_10283_2325':
        return get_DS_10283_2325
    elif dataset_type == 'Audiostock_splited':
        return get_Audiostock_splited
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, model_cfg):
    data = {}

    args.class_index_dict = load_class_label(args.class_label_path)

    if args.datasetinfos is None:
        args.datasetinfos = ["train", "unbalanced_train", "balanced_train"]
    if args.dataset_type == "webdataset":
        args.train_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            args.datasetinfos,
            islocal=not args.remotedata,
            proportion=args.dataset_proportion,
            dataset_path=args.datasetpath,
            full_dataset=args.full_train_dataset,
        )

        if args.full_train_dataset is None:
            args.full_train_dataset = []
        if args.exclude_eval_dataset is None:
            args.exclude_eval_dataset = []
        excluded_eval_datasets = args.full_train_dataset + args.exclude_eval_dataset

        val_dataset_names = [n for n in args.datasetnames if n not in excluded_eval_datasets] \
            if excluded_eval_datasets else args.datasetnames
        args.val_dataset_names = val_dataset_names
        args.val_data = get_tar_path_from_dataset_name(
            val_dataset_names,
            ["valid", "test", "eval"],
            islocal=not args.remotedata,
            proportion=1,
            dataset_path=args.datasetpath,
            full_dataset=None,
        )

    if args.dataset_type == "Audiostock-10k-16khz":
        assert len(args.datasetnames) == len(args.datasetinfos), "datasetnames datasetinfos must have equal sizes."
        files = []
        # dataset_names = []
        for root, _, file_names in os.walk(args.datasetpath):
            for file_name, dataset_info in zip(args.datasetnames, args.datasetinfos):
                file_path = os.path.join(root, file_name + ".txt")
                if os.path.exists(file_path):
                    files.append((file_path, os.path.basename(os.path.dirname(os.path.dirname(file_path))), dataset_info))

        
        args.train_data = [pair[0] for pair in files if pair[2] == 'train']
        args.val_data = [pair[0] for pair in files if pair[2] == 'valid']
        args.val_dataset_names = [pair[1] for pair in files if pair[2] == 'valid']

    if args.dataset_type == "DS_10283_2325":
        assert len(args.datasetnames) == len(args.datasetinfos), "datasetnames datasetinfos must have equal sizes."
        files = []
        # dataset_names = []
        for root, _, file_names in os.walk(args.datasetpath):
            for file_name, dataset_info in zip(args.datasetnames, args.datasetinfos):
                file_path = os.path.join(root, file_name + ".json")
                if os.path.exists(file_path):
                    files.append((file_path, os.path.basename(os.path.dirname(file_path)), dataset_info))

        
        args.train_data = [pair[0] for pair in files if pair[2] == 'train']
        args.val_data = [pair[0] for pair in files if pair[2] == 'valid']
        args.val_dataset_names = [pair[1] for pair in files if pair[2] == 'valid']

    if args.dataset_type == "Audiostock_splited":
        assert len(args.datasetnames) == len(args.datasetinfos), "datasetnames datasetinfos must have equal sizes."
        files = []
        # dataset_names = []
        for root, _, file_names in os.walk(args.datasetpath):
            for file_name, dataset_info in zip(args.datasetnames, args.datasetinfos):
                file_path = os.path.join(root, file_name + ".json")
                if os.path.exists(file_path):
                    files.append((file_path, os.path.basename(os.path.dirname(file_path)), dataset_info))

        
        args.train_data = [pair[0] for pair in files if pair[2] == 'train']
        args.val_data = [pair[0] for pair in files if pair[2] == 'valid']
        args.val_dataset_names = [pair[1] for pair in files if pair[2] == 'valid']

    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, model_cfg, is_train=True
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.dataset_type)(
            args, model_cfg, is_train=False
        )

    return data

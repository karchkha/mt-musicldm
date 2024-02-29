import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.latent_diffusion.util import instantiate_from_config
from utilities.data.dataset import AudiostockDataset, DS_10283_2325_Dataset
import torch
import omegaconf




class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size=2, num_workers=1, augmentation= None, path=None, preprocessing=None, config = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers # if num_workers is not None else batch_size*2
        self.path = path
        self.augmentation=augmentation
        self.preprocessing=preprocessing
        
        self.config = {}
        self.config["path"] = path
        self.config["preprocessing"] = preprocessing
        self.config["augmentation"] = augmentation


    def prepare_data(self):
        # This method is used for data download and preprocessing (if any)
        # It is called only once across all GPUs in a distributed setup
        # You can use this method to download the data or perform any necessary preprocessing steps.
        pass       

    def setup(self, stage=None):

        if 'train_data' in self.path and self.path['train_data'] is not None:
            self.train_dataset = self.load_dataset(self.path['train_data'], split = "train")

            if hasattr(self.path, 'split') and self.path['split']:
                if self.path['split'] == "test":
                    self.train_dataset, self.test_dataset = self.split_data(self.train_dataset) #sometime we want opur split to be test for some technical reasons
                else:
                    self.train_dataset, self.val_dataset = self.split_data(self.train_dataset)
        
        if 'valid_data' in self.path and self.path['valid_data'] is not None:
            self.val_dataset  = self.load_dataset(self.path['valid_data'], split = "valid") 
        
        if 'test_data' in self.path and self.path['test_data'] is not None:
            self.test_dataset = self.load_dataset(self.path['test_data'], split = "test") 

        if not hasattr(self, 'train_dataset') and not hasattr(self, 'val_dataset') and not hasattr(self, 'test_dataset'):
            raise ValueError("Invalid dataset configuration provided.")

    # def get_data_handler(self):
    #     if self.path["data_handler"] == "Dcase":
    #         return Dcase_Dataset
    #     elif self.path["data_handler"] == "WavCaps":
    #         return WavCaps_Dataset
    #     if self.path["data_handler"] == "Clotho":
    #         return Clotho_Dataset
        
    #     # Add other types of data here!
    #     else:
    #         raise ValueError(f"Unsupported data format: {self.data_format}")

    def get_data_handler(self, path):
        keywords = ['Audiostock', 'DS_10283_2325']  # Keywords to identify data handlers
        handler = None
        if type(path) is list or type(path) is omegaconf.listconfig.ListConfig:
            
            print("Attention!!! You are mixing datasets. Please remember to only mix datasets from the same data group by each 'train', 'val', 'test' splits.")
            detected_keywords = []

            for p in path:
                for keyword in keywords:
                    if keyword in p:
                        if not any(keyword not in detected_keyword for detected_keyword in detected_keywords):
                            detected_keywords.append(keyword)
                        else:
                            raise ValueError("you mixed more that one data group in one split.")
            path = path[0]


        for keyword in keywords:
            if keyword in path:
                if keyword == 'Audiostock':
                    handler = AudiostockDataset
                elif keyword == 'DS_10283_2325':
                    handler = DS_10283_2325_Dataset

                print(f"Data format '{keyword}' detected. Using {handler.__name__} as the data handler for: {path} ")
        if handler is None:
            raise ValueError(f"Unsupported data format: {path}")
        else:
            return handler 


    def load_dataset(self, path, split = "train"):

        dataset_subclass = self.get_data_handler(path)
        if split == "train":
            return dataset_subclass( 
                                dataset_path=path,
                                label_path=self.config["path"]["label_data"],
                                config=self.config,
                                train=True,
                                factor=1.0        
                                )
        else:
            return dataset_subclass( 
                                dataset_path=path,
                                label_path=self.config["path"]["label_data"],
                                config=self.config,
                                train=False,
                                factor=1.0        
                                )
                            # preprocess_config = self.preprocessing,  
                            # train_config = self.augmentation, 
                            # path =  path, 
                            # class_label_index = self.path['class_label_index'] if hasattr(self.path, 'class_label_index') else None, 
                            # split = split
                            # )
    #     dataset = AudiostockDataset(
    #             dataset_path=config["path"]["train_data"],
    #             label_path=config["path"]["label_data"],
    #             config=config,
    #             train=True,
    #             factor=1.0
    #         )
    def split_data(self, dataset):
        # Split the dataset into train, validation, and test sets
        train_len = int(0.9 * len(dataset))
        val_len = int(0.1 * len(dataset))

        train_dataset, val_dataset = torch.utils.data.random_split(
                                    dataset, [train_len, val_len], 
                                    generator=torch.Generator().manual_seed(42))
        print(f"Dataset has been splitted! Train: {len(train_dataset)}, Valid: {len(val_dataset)}")

        return train_dataset, val_dataset

    def train_dataloader(self):
        # Returns the DataLoader for the training dataset
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # Returns the DataLoader for the validation dataset
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Returns the DataLoader for the test dataset
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
import glob
import os
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np
from .transforms import compose
import pandas as pd
from PIL import Image
import yaml

data_folder =  '/home/ibrahimalmakky/Documents/data/ISIC_2019'
csv_file = "ISIC_2019_datasplit.csv"
imgs_folder = "ISIC_2019_Training_Input"
def get_transforms(mode="train"):
    # preprocessing
    return compose(
        yaml.load(
            open('/home/santoshsanjeev/MedMerge/params/transforms/rsna_pneumonia.yaml', encoding="UTF-8"), Loader=yaml.FullLoader
        )[mode]
    )



class ISIC_2019(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.loc[self.data_frame["split"] == split]
        self.root_dir = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 1]
        img_path = f"{self.root_dir}/{img_name}.jpg"
        image = Image.open(img_path)

        label = self.data_frame.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_train_loaders(config):
    train_dataset = config['wrapper'](
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="train",
        transform=get_transforms(mode="train"),
    )

    train_loader = {'full': torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_train'],
        num_workers=config['num_workers'],
    )}

    return train_loader


def prepare_val_loaders(config):
    val_dataset = config['wrapper'](
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="val",
        transform=get_transforms(mode="test"),
    )

    
    val_loader = {'full':torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_test'],
        num_workers=config['num_workers'],
    )}


    return val_loader





def prepare_test_loaders(config):
    test_dataset = config['wrapper'](
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="test",
        transform=get_transforms(mode="test"),
    )

    test_loader = {'full':torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_test'],
        num_workers=config['num_workers'],
    )}

    test_loader['class_names'] = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion', 'Squamous Cell Carcinoma']
    return test_loader   
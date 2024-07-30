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

data_folder= '/home/ibrahimalmakky/Documents/data/rsna_18'
csv_file= '/home/ibrahimalmakky/Documents/data/rsna_18/csv/final_dataset_wo_not_normal_cases.csv'
  

def get_transforms(mode="train"):
    # preprocessing
    return compose(
        yaml.load(
            open('/home/santoshsanjeev/MedMerge/params/transforms/rsna_pneumonia.yaml', encoding="UTF-8"), Loader=yaml.FullLoader
        )[mode]
    )



class RSNADataset(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.split = split
        # self.data = self.data[self.data['class'] != 'No Lung Opacity / Not Normal']

        # Filter the data based on the specified split
        self.data = self.data[self.data["split"] == split]
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.data_folder, "train")
        image_path = os.path.join(image_folder, self.data.iloc[idx, 1] + ".jpg")
        image = Image.open(image_path).convert("RGB")

        target = int(self.data.iloc[idx, 7])
        if self.transform:
            image = self.transform(image)
        return image, target


def prepare_train_loaders(config):
    train_dataset = config['wrapper'](
        csv_file = csv_file, 
        data_folder = data_folder, 
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
        csv_file=csv_file,
        data_folder = data_folder, 
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
        csv_file= csv_file, 
        data_folder=data_folder, 
        split="test",
        transform=get_transforms(mode="test"),
    )

    test_loader = {'full':torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_test'],
        num_workers=config['num_workers'],
    )}

    test_loader['class_names'] = ['Positive', 'Negative']
    return test_loader
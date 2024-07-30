import os
import torch
import random
import sys
sys.path.append('./')  # Add the parent directory to the Python path

from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np

from utils import *
from model_merger import ModelMerge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import glob
import os
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np

import pandas as pd
from PIL import Image

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
"""
Author: Ibrahim Almakky
Date: 13/02/2022
"""
import yaml
from torchvision import transforms


def random_horizontal_flip(params: dict):
    return transforms.RandomHorizontalFlip(**params)


def random_rotation(params: dict):
    return transforms.RandomRotation(**params)


def gaussian_blur(params: dict):
    return transforms.GaussianBlur(**params)


def resize(params: dict):
    return transforms.Resize(**params)


def random_scale(params: dict):
    return transforms.RandomResizedCrop(**params)


def normalize(params: dict):
    return transforms.Normalize(**params)


def to_tensor(params: dict):
    return transforms.ToTensor(**params)

TRANSFORMS = {
    "horizontal_flip": random_horizontal_flip,
    "random_rotation": random_rotation,
    "gaussian_blur": gaussian_blur,
    "resize": resize,
    "random_scale": random_scale,
    "normalize": normalize,
    "to_tensor": to_tensor,
}


def compose(transforms_strs: dict):
    """
    Input images are assumed to be tensors
    """
    transforms_list = []
    for name, params in transforms_strs.items():
        assert name in TRANSFORMS.keys()
        transforms_list.append(TRANSFORMS[name](params))
    transforms_composed = transforms.Compose(transforms_list)
    return transforms_composed

def get_transforms(mode="train"):
    # preprocessing
    return compose(
        yaml.load(
            open('/home/santoshsanjeev/MedMerge/params/transforms/rsna_pneumonia.yaml', encoding="UTF-8"), Loader=yaml.FullLoader
        )[mode]
    )


def prepare_ISIC_2019():

        # data_folder: /home/ibrahimalmakky/Documents/data/ISIC_2019
    # num_classes: 8
    # num_workers: 8
    # transforms: ./params/transforms/rsna_pneumonia.yaml
    batch_size = 32
    data_folder = '/home/ibrahimalmakky/Documents/data/ISIC_2019/'
    num_workers = 8

    csv_file = "ISIC_2019_datasplit.csv"
    imgs_folder = "ISIC_2019_Training_Input"

    train_dataset = ISIC_2019(
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="train",
        transform=get_transforms(mode="train"),
    )

    val_dataset = ISIC_2019(
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="val",
        transform=get_transforms(mode="test"),
    )

    test_dataset = ISIC_2019(
        csv_file=os.path.join(data_folder, csv_file), 
        data_folder=os.path.join(data_folder, imgs_folder), 
        split="test",
        transform=get_transforms(mode="test"),
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def run_auxiliary_experiment(merging_fn, experiment_config, pairs, device, stop_at=None, csv_file=''):
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(experiment_config)
        # train_loader = config['data']['train']['full']
        train_loader, val_loader, test_loader = prepare_ISIC_2019()

        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        
        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(
            deepcopy(config['models']['new']), 
            train_loader, 
            transform_fn=get_merging_fn(merging_fn), 
            metric_classes=config['metric_fns'],
            stop_at=stop_at,
        )
        reset_bn_stats(Merge, train_loader)
        
        results = evaluate_model(experiment_config['eval_type'], Merge, config, test_loader)
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = merging_fn
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = split
        write_to_csv(results, csv_file=csv_file)
        print(results)
        
    return results


if __name__ == "__main__":
    # config_name = 'cifar5_vgg'
    config_name = 'isic2019_resnet50'
    skip_pair_idxs = []
    merging_fns = [
        'match_tensors_zipit',
        'match_tensors_permute',
        'match_tensors_identity',
    ]
    stop_at = None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    # run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    run_pairs = [['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_04-59-43_BioMedIA-A5000ResNet50_HAM10K_to_ISIC_FT_No_LP/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_06-17-07_BioMedIA-A5000ResNet50_ImageNet_to_ISIC_FT_No_LP/last.ckpt']]#find_runable_pairs(model_dir, model_name, skip_pair_idxs=None)

    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'auxiliary_functions_v2.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with torch.no_grad():
        for merging_fn in merging_fns:
            node_results = run_auxiliary_experiment(
                merging_fn=merging_fn, 
                experiment_config=raw_config, 
                pairs=run_pairs, 
                device=device, 
                csv_file=csv_file,
                stop_at=stop_at
            )
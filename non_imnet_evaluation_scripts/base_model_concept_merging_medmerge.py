# import os
# import torch
# import random
# from time import time
# from copy import deepcopy
# import torchvision
# import torchvision.transforms as T
# from tqdm.auto import tqdm
# import numpy as np
# import sys
# sys.path.append('./')  # Add the parent directory to the Python path

# from utils import *

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

    

# def evaluate_pair_models(eval_type, models, config, csv_file):
#     # train_loader = config['data']['train']['full']
#     data_dir = './data/cifar-10-python'
#     wrapper = torchvision.datasets.CIFAR10
#     num_classes = 10  
#     batch_size = 500                     # num classes in dataset   
#     CIFAR_MEAN = [125.307, 122.961, 113.8575]
#     CIFAR_STD = [51.5865, 50.847, 51.255]
#     normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
#     denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))


#     train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
#     train_dset = wrapper(root=data_dir, train=True, download=True, transform=train_transform)
    
#     train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=8)
        
#     print([base_model for base_model in config['models']['bases']])
#     models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
#     models = config['models']['bases']
    
#     for idx, model in enumerate(models):
#         results = evaluate_model(eval_type, model, config)
#         results['Model'] = CONCEPT_TASKS[idx]
#         write_to_csv(results, csv_file=csv_file)
#         print(results)
    
#     ensembler = SpoofModel(models)
#     results = evaluate_model(eval_type, ensembler, config)
#     results['Model'] = 'Ensemble'
#     write_to_csv(results, csv_file=csv_file)
#     print(results)


# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     config_name = 'cifar10_resnet50'
#     skip_pair_idxs = []
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     raw_config = get_config_from_name(config_name, device=device)
#     model_dir = raw_config['model']['dir']
#     model_name = raw_config['model']['name']
#     print(model_dir, model_name)
#     run_pairs = ['./checkpoints/resnet50_v0.pth.tar','./checkpoints/resnet50_v1.pth.tar']#find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
#     csv_file = os.path.join(
#         './csvs',
#         raw_config['dataset']['name'],
#         raw_config['model']['name'],
#         raw_config['eval_type'],
#         'base_models.csv'
#     )
#     print(run_pairs, len(run_pairs))
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)    

#     with torch.no_grad():
#         for pair in run_pairs:
#             # raw_config = inject_pair(raw_config, pair)
#             config = prepare_experiment_config(raw_config)

#             evaluate_pair_models(
#                 eval_type=config['eval_type'],
#                 models=config['models']['bases'],
#                 config=config,
#                 csv_file=csv_file
#             )


import sys
sys.path.append('./')
import os
import torch
import random
from time import time
from copy import deepcopy

from tqdm.auto import tqdm
import numpy as np

from utils import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

    

def evaluate_pair_models(eval_type, models, config, csv_file, split):
    print(split)
    train_loader = config['data']['train']['full']
    models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
    models = config['models']['bases']
    
    for idx, model in enumerate(models):
        results, model = evaluate_model(eval_type, model, config, split)
        breakpoint()
        results['Model'] = CONCEPT_TASKS[idx]
        write_to_csv(results, csv_file=csv_file)
        print(results)
    
    # ensembler = SpoofModel(models)
    # results = evaluate_model(eval_type, ensembler, config)
    # results['Model'] = 'Ensemble'
    # write_to_csv(results, csv_file=csv_file)
    # print(results)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = 'isic2019_densenet'
    # skip_pair_idxs = [0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    # run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    run_pairs = [
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_06-17-07_BioMedIA-A5000ResNet50_ImageNet_to_ISIC_FT_No_LP/last.ckpt', '/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_04-59-43_BioMedIA-A5000ResNet50_HAM10K_to_ISIC_FT_No_LP/last.ckpt'], #ISIC_FT_NO_LP_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar07_23-52-50_BioMedIA-A5000ResNet50_ImageNet_to_ISIC_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_06-18-38_BioMedIA-A5000ResNet50_HAM_to_ISIC_Full_FineTuning/last.ckpt'], #ISIC_Full_finetuning_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar07_23-53-41_BioMedIA-A5000ResNet50_ImageNet_and_HAM_to_ISIC_WeightedKernelAvgModel_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar07_23-53-41_BioMedIA-A5000ResNet50_ImageNet_and_HAM_to_ISIC_WeightedKernelAvgModel_Full_FineTuning/last.ckpt'] #ISIC_Medmerge_ResNet50
        
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_04-36-54_BioMedIA-A5000ResNet50_ImageNet_to_Aptos_FT_No_LP/last.ckpt', '/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_06-13-44_BioMedIA-A5000ResNet50_EyePACS_to_APTOS_my_model_FT_No_LP/last.ckpt'], #APTOS_FT_NO_LP_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_04-46-33_BioMedIA-A5000ResNet50_ImageNet_to_Aptos_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_06-09-48_BioMedIA-A5000ResNet50_EyePACS_to_APTOS_my_model_Full_FineTuning/last.ckpt'] #APTOS_Full_finetuning_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_06-07-50_BioMedIA-A5000ResNet50_ImageNet_and_EyePACS_to_Aptos_WeightedKernelAvgModel_my_model_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_06-07-50_BioMedIA-A5000ResNet50_ImageNet_and_EyePACS_to_Aptos_WeightedKernelAvgModel_my_model_Full_FineTuning/last.ckpt'] #APTOS_Medmerge_ResNet50
        
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_07-37-53_BioMedIA-A5000ResNet50_ImageNet_to_RSNA_FT_No_LP/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_07-38-44_BioMedIA-A5000ResNet50_CheXpert_to_RSNA_FT_No_LP/last.ckpt'], #RSNA_FT_NO_LP_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_06-21-45_BioMedIA-A5000ResNet50_ImageNet_to_RSNA_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_06-22-31_BioMedIA-A5000ResNet50_CheXpert_to_RSNA_Full_FineTuning/last.ckpt'], #RSNA_Full_finetuning_ResNet50
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_07-36-50_BioMedIA-A5000ResNet50_ImageNet_and_CheXpert_to_RSNA_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_07-36-50_BioMedIA-A5000ResNet50_ImageNet_and_CheXpert_to_RSNA_Full_FineTuning/last.ckpt'] #RSNA_medmerge_ResNet50
        
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_09-08-31_BioMedIA-A5000DenseNet121_ImageNet_to_ISIC19_FT_No_LP/last.ckpt', '/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_09-04-03_BioMedIA-A5000DenseNet121_HAM10K_to_ISIC19_FT_No_LP/last.ckpt'],
        ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_11-08-48_BioMedIA-A5000DenseNet121_ImageNet_to_ISIC19_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_14-51-00_BioMedIA-A5000DenseNet121_HAM10K_to_ISIC19_Full_FineTuning/last.ckpt'],        
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_13-45-10_BioMedIA-A5000DenseNet121_ImageNet_and_HAM10K_to_ISIC19_WeightedKernelAvgModel_Full_FineTuning/last.ckpt', '/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_13-45-10_BioMedIA-A5000DenseNet121_ImageNet_and_HAM10K_to_ISIC19_WeightedKernelAvgModel_Full_FineTuning/last.ckpt']

        # ['',''],
        # ['',''],
        # ['',''],

        # ['',''],
        # ['',''],
        # ['',''],
        ]

    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'base_models_medmerge.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)    

    with torch.no_grad():
        for pair in run_pairs:
            raw_config = inject_pair(raw_config, pair)
            config = prepare_experiment_config(raw_config)

            evaluate_pair_models(
                eval_type=config['eval_type'],
                models=config['models']['bases'],
                config=config,
                csv_file=csv_file,
                split = list(range(0,config['model']['output_dim'])),
            )
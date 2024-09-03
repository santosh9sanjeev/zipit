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


def run_auxiliary_experiment(merging_fn, experiment_config, pairs, device, split, stop_at=None, csv_file=''):
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(experiment_config)
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        # base_models = [base_model for base_model in config['models']['bases']]

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
        # Merge.merged_model.classifier.weight = base_models[0].classifier.weight
        # Merge.merged_model.features.norm5 = base_models[0].features.norm5

        print("Do the layer weights match?")
        for model_layer, merged_layer in zip(base_models[0].named_parameters(), 
                                             Merge.merged_model.named_parameters()):
            model_layer_name, model_layer_params = model_layer[0], model_layer[1]
            print(model_layer_name)
            print(torch.all(model_layer_params == merged_layer[1]))
        
        results, model = evaluate_model(experiment_config['eval_type'], Merge, config, split)
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = merging_fn
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = split
        print(results)
        write_to_csv(results, csv_file=csv_file)
        
    return results, model


if __name__ == "__main__":
    # config_name = 'cifar5_vgg'
    config_name = 'isic2019_vgg'
    # config_name = 'rsna_densenet'
    # config_name = 'aptos_densenet'
    skip_pair_idxs = []
    merging_fns = [
        'match_tensors_zipit',
        'match_tensors_permute',
        'match_tensors_identity',
    ]
    stop_at = 48
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    # run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    run_pairs = [
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar07_23-52-50_BioMedIA-A5000ResNet50_ImageNet_to_ISIC_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar07_23-52-50_BioMedIA-A5000ResNet50_ImageNet_to_ISIC_Full_FineTuning/last.ckpt']#'/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar08_06-18-38_BioMedIA-A5000ResNet50_HAM_to_ISIC_Full_FineTuning/last.ckpt'], #ISIC_Full_finetuning
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_04-46-33_BioMedIA-A5000ResNet50_ImageNet_to_Aptos_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar08_06-09-48_BioMedIA-A5000ResNet50_EyePACS_to_APTOS_my_model_Full_FineTuning/last.ckpt'] #APTOS_Full_finetuning
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_06-21-45_BioMedIA-A5000ResNet50_ImageNet_to_RSNA_Full_FineTuning/last.ckpt','/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Mar08_06-22-31_BioMedIA-A5000ResNet50_CheXpert_to_RSNA_Full_FineTuning/last.ckpt'], #RSNA_Full_finetuning

        # DenseNets
        # ISIC2019
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_11-08-48_BioMedIA-A5000DenseNet121_ImageNet_to_ISIC19_Full_FineTuning/last.ckpt',
        #  '/home/santoshsanjeev/MedMerge/logs/train_val/ISIC19/Mar05_14-51-00_BioMedIA-A5000DenseNet121_HAM10K_to_ISIC19_Full_FineTuning/last.ckpt'
        #  ],
        # 
        # RSNA
        # ['/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Feb28_09-26-48_ip-140-0-8-142DenseNet121_ImageNet_to_RSNA_no_img_norm_Full_FineTuning/last.ckpt',
        #  '/home/santoshsanjeev/MedMerge/logs/train_val/rsna/Feb29_05-13-07_ip-140-0-8-142DenseNet121_Chexpert_to_RSNA_Full_FineTuning/last.ckpt',
        #  ]   

        # APTOS
        # [
            # '/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar05_11-10-57_BioMedIA-A5000DenseNet121_ImageNet_to_APTOS_Full_FineTuning/last.ckpt',
            # '/home/santoshsanjeev/MedMerge/logs/train_val/aptos/Mar05_15-58-38_BioMedIA-A5000DenseNet121_EyePACS_to_Aptos_Full_FineTuning/last.ckpt',
        # ]
        
        #ISIC VGG
        [   '/home/ibrahimalmakky/Documents/projects/MedMerge/logs/train_val/ISIC19/Aug26_15-17-53_BioMedIA-A5000VGG16_ImageNet_to_ISIC19_Full_FineTuning/last.ckpt',
            '/home/ibrahimalmakky/Documents/projects/MedMerge/logs/train_val/ISIC19/Aug26_13-36-10_BioMedIA-A5000VGG16_HAM10K_to_ISIC19_Full_FineTuning/last.ckpt',
        ]
        ]
        
        #find_runable_pairs(model_dir, model_name, skip_pair_idxs=None)

    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'auxiliary_functions_medmerge_vgg.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with torch.no_grad():
        for merging_fn in merging_fns:
            node_results, model = run_auxiliary_experiment(
                merging_fn=merging_fn, 
                experiment_config=raw_config, 
                pairs=run_pairs, 
                device=device, 
                split = list(range(0,raw_config['model']['output_dim'])),
                csv_file=csv_file,
                stop_at=stop_at
            )
            torch.save(model.state_dict(), f"./auxiliary_models/{raw_config['dataset']['name']}_{raw_config['model']['name']}_{merging_fn}.pt")
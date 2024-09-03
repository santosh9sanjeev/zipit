config = {
    'dataset': {
        'name': "isic2019",               # name of the dataset. Should match a corresponding variable name found in datasets/config.py
    },
    'model': {
        'name': 'vgg16',                 # name of the model
        'dir': '',              # checkpoint directory where models are stored
        'bases': [],
        'output_dim':8,                             # list of optional model paths. Empty by default
    },

    'merging_fn': 'match_tensors_zipit',        # matching function desired. Please see "matching_functions.py" for a complete list of supported functions.
    'eval_type': 'logits',                        # Evaluation type, whether to use clip or standard cross entropy loss
    'merging_metrics': ['covariance', 'mean'],  # Alignment Metric types desired upon which to compute merging. Please see metric_calculators.py for more details
}
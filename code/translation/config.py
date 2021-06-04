config = {
    'image_path': 'samples', #'datasets/bdd100k' # expects .png
    'ckpt_path': 'ckpt',
    'ckpt_name': None,
    'batch_sizes': (16, 24, 1),
    'max_epochs': 20,
    'num_workers': 40,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': True,
    'shuffle': (True, True, True)
}

debug_options = {
        'image_path': '../../samples/test',
}

log_keys = [
        'model_name',
]


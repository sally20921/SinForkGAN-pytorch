config = {
    'image_path': '../../test', #'../../datasets/bdd100k'
    'ckpt_path': '../../ckpt',
    'ckpt_name': None,
    'batch_sizes': (16, 24, 3),
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
        'image_path': '../../test',
}

log_keys = [
        'model_name',
]


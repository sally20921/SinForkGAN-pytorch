config = {
    'image_path': 'samples', #'datasets/bdd100k' # expects .png
    'ckpt_path': 'ckpt',
    'ckpt_name': None,
    'batch_sizes': (1, 1, 1),
    'max_epochs': 200,
    'num_workers': 1,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'lambdaA': 10,
    'lambdaB': 10,
    'beta1': 0.5,
    'beta2': 0.999,
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': True,
    'shuffle': (True, True, True),
    'model_name': 'sfg'.
    'loss_name': 'cross_entropy',
}

debug_options = {
        'image_path': '../../samples',
}

log_keys = [
        'model_name',
]


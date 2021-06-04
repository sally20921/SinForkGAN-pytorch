import os
from pathlib import Path
from torch import optim
from inflection import underscore

optim_dict = {}

def add_optims():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, "__bases__"):
                    optim_dict[underscore(str(member.__name__))] = member

def get_optimizer(args, model):
    optim = optim_dict[args.optimizer]
    optim = optim.resolve_args(args, model.parameters())
    optim.zero_grad()
    return optim

add_optims()
print(optim_dict)


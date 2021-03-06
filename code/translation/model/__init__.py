import os
from pathlib import Path
from torch import nn
from inflection import underscore

model_dict = {}

def add_models():
    path = Path(os.path.dirname(__file__))
    print(path)

    for p in path.glob('*.py'):
        name  = p.stem
        print("name: ", name)
        parent = p.parent.stem
        print("parent: ", parent)
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                # add to dict all nn.Module classes
                if hasattr(member, "__bases__") and nn.Module in member.__bases__:
                            model_dict[underscore(str(member.__name__))] = member

def get_models(args, model_name):
    model  = model_dict[model_name]
    #model = model.resolve_args(args)
    return model.to(args.device)

add_models()
print("model dictionary: ", model_dict)


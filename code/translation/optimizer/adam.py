from torch import optim

class Adam(optim.Adam):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.0001)
        options['weight_decay'] = args.get("weight_decay", 1e-5)
        return cls(params, **options)

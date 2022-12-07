import torch

class Hook_var():
    def __init__(self):
        self._feature = None

    def __call__(self, module, input, output):

        output = torch.squeeze(output).cpu().data.numpy()
        self._feature = output

    def clear(self):
        self._feature = None

    def get_feature(self):

        return self._feature

def register_hooks(ly_names,net):
    modules = dict((name, module) for name, module in net.named_modules())
    ly_hooks = dict()
    for ly_name in ly_names:
        ly_hooks[ly_name] = Hook_var()
        if ly_name in modules:
            modules[ly_name].register_forward_hook(ly_hooks[ly_name])
    return ly_hooks

def extract_hook_features(hooks):

    features = dict()

    for name, hook in hooks.items():
        features[name] = hook.get_feature()
    return features
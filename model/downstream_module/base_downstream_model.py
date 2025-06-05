seg_modules = {}

def register_seg_module(name):
    def decorator(cls):
        seg_modules[name] = cls
        return cls
    return decorator

def get_seg_module(name):
    return seg_modules[name]
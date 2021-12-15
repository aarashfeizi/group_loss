from .cub import Birds
from .hotels import Hotels

_type = {
    'cars': Birds,
    'cub': Birds,
    'Stanford': Birds,
    'hotels': Hotels
}


def load(name, root, mode, transform=None, labels=None, is_extracted=False):
    if name == 'hotels':
        return _type[name](root=root, mode=mode, transform=transform)
    else:
        return _type[name](root=root, labels=labels, is_extracted=is_extracted,  transform=transform)

import torch
import random
import argparse
import numpy as np


def customize_seed(seed):
    """
    Sets seeds across libraries to ensure reproducible results.

    Args:
        seed (int): The seed value to be used for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Uncomment below if using multi-GPU setup
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
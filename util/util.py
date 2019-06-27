import numpy as np
import torch
import pathlib


def config(seed):
    np.random.seed(7)
    torch.manual_seed(7)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

import torch
import os
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
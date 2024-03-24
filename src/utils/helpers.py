import torch
import random
import numpy as np

def count_finetuned_params(model):
    num_finetuned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_finetuned_params

def count_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import random
import numpy as np
import torch
import time
from thop import profile
from torch.cuda.amp import autocast


def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # if use GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_flops_params(model, input):
    '''
    Calculate the FLOPs and parameters of the model.
    '''
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"FLOPs: {flops/1e9:.4f} G, Params: {params/1e6:.4f} M")

def calculate_runtime(model, input_size, input = None, batch_size = 1, iter = 30, device = 'cuda', use_amp=False, need_flops=False):
    '''
    Calculate the runtime of the model.
    '''
    model.to(device); model.eval()
    if input is None:
        input = torch.randn(batch_size, *input_size).to(device)
    start_time = time.time()
    for i in range(iter):
        with torch.no_grad():
            if use_amp:
                with autocast():
                        model(input)
            else:
                    model(input)
    end_time = time.time()
    if need_flops:
        calculate_flops_params(model, input)
    runtime = (end_time - start_time) / iter
    fps = batch_size / runtime
    print(f"Runtime: {runtime*1000:.4f} ms, FPS: {fps:.4f}")
    
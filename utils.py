import torch
from torch import nn
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2 as irfft
    from torch.fft import rfft2  as rfft

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

def hit_at_k(predictions, gt_idx, k = 10):
    assert predictions.shape[0] == gt_idx.shape[0]
    zero_tensor = torch.tensor([0], device = predictions.device)
    one_tensor = torch.tensor([1], device = predictions.device)
    _, indices = predictions.topk(k = k, largest = False)
    return torch.where(indices == gt_idx, one_tensor, zero_tensor).sum().item()

def mrr(predictions, gt_idx):
    indices = predictions.argsort()
    return (1.0 / (indices == gt_idx).nonzero()[:, 1].float().add(1.0)).sum().item()

def com_mult(a, b):
    r1, i1 = a[:, 0], a[:, 1]
    r2, i2 = b[:, 0], b[:, 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)
    
def conj(a): 
    #print("in conj", a.shape)
    a[:, 1] = -a[:, 1]
    return a
def ccorr(a, b):
    #print("in ccorr", a.shape)
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)),a.shape[-1])
import torch

def is_cuda_tensor(t):
    assert torch.is_tensor(t)
    assert t.is_cuda

def check_shape_len(t, n):
    assert len(t.shape) == n

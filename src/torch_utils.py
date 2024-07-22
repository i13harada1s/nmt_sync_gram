import torch


def cumsum(tensor, dim, reverse=True):
    """Wrapper of troch.cumsum()"""
    if reverse:
        tensor = tensor.flip([-1]).cumsum(dim, dtype=torch.float32).flip([-1])
    else:
        tensor = tensor.cumsum(dim, dtype=torch.float32)
    return tensor

def cumprod(tensor, dim, reverse=True):
    """Wrapper of troch.cumprod()"""
    if reverse:
        tensor = tensor.flip([-1]).cumprod(dim, dtype=torch.float32).flip([-1])
    else:
        tensor = tensor.cumprod(dim, dtype=torch.float32)
    return tensor

def cummax(tensor, dim, reverse=True):
    """Wrapper of troch.cummax()"""
    if reverse:
        tensor = tensor.flip([-1]).cummax(dim)[0].flip([-1])
    else:
        tensor = tensor.cummax(dim)[0]
    return tensor

def cummin(tensor, dim, reverse=True):
    """Wrapper of troch.cummin()"""
    if reverse:
        tensor = tensor.flip([-1]).cummin(dim)[0].flip([-1])
    else:
        tensor = tensor.cummin(dim)[0]
    return tensor

def shift(tensor, shift: int, dim: int, footmark: float=0.0):
    length = tensor.size(dim)
    if abs(shift) > length:
        return tensor.fill_(footmark)
    if shift < 0:
        index = torch.arange(length+shift, length, device=tensor.device)
    else:
        index = torch.arange(0, shift, device=tensor.device)
    return tensor.roll(shift, dim).index_fill(dim, index, footmark)

def sigmoid(x, onnx_trace: bool=False):
    if onnx_trace:
        return torch.sigmoid(x.float())
    else:
        return torch.sigmoid(x.to(dtype=torch.float32))

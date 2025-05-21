import numpy as np
import torch

# Handle default device placement of tensors and float precision
_device = 'cpu'
def set_torch_device(device):
    global _device
    _device = device
    update_torch_default_tensor_type()
def get_torch_device():
    return _device


def update_torch_default_tensor_type():
    if get_torch_device() == 'cpu':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'cuda':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'cuda:1':
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    elif get_torch_device() == 'xpu':
        import torch_ipex
        if get_float_dtype() == np.float32:
            torch.set_default_tensor_type(torch.xpu.FloatTensor)
        elif get_float_dtype() == np.float64:
            torch.set_default_tensor_type(torch.xpu.DoubleTensor)
        else:
            raise NotImplementedError(f'Unknown float dtype {get_float_dtype()}')
    else:
        raise NotImplementedError(f'Unknown device {get_torch_device()}')


_float_dtype = np.float32
def get_float_dtype():
    return _float_dtype
def set_float_dtype(np_dtype):
    global _float_dtype
    _float_dtype = np_dtype
    update_torch_default_tensor_type()
def get_float_torch_dtype():
    if _float_dtype == np.float32:
        return torch.float32
    elif _float_dtype == np.float64:
        return torch.float64
    else:
        raise NotImplementedError(f'unknown np dtype {_float_dtype}')
def get_complex_torch_dtype():
    if _float_dtype == np.float32:
        return torch.complex64
    elif _float_dtype == np.float64:
        return torch.complex128
    else:
        raise NotImplementedError(f'unknown np dtype {_float_dtype}')
def set_float_prec(prec):
    if prec == 'single':
        set_float_dtype(np.float32)
    elif prec == 'double':
        set_float_dtype(np.float64)
    else:
        raise NotImplementedError(f'Unknown precision type {prec}')
def init_device():
    if torch.cuda.is_available():
        set_torch_device('cuda')
        return 'cuda'
    try:
        import torch_ipex
        if torch.xpu.is_available():
            set_torch_device('xpu')
            return 'xpu'
    except:
        pass
    return 'cpu'
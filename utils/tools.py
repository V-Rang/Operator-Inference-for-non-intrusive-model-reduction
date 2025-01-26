import torch
import numpy as np

def ckron_tensor(arr):
    return torch.cat([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

def ckron_numpy(arr):
    return np.concatenate([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)


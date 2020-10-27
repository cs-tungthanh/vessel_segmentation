import os, torch
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from .data_utils import Helper
torch.set_default_dtype(torch.float32)

class Dataset2D(Dataset):
    def __init__(self, path, eof_name, transforms=None, loader_fnc=Helper.loadTensor):
        super().__init__()
        self.path = path
        self.loader = loader_fnc # Load tensor with type of Float32

        print('Loading Dataset')
        self._volume = self.loader(path + '/vol_' + eof_name + '.pth')
        self._mask = self.loader(path + '/mask_' + eof_name + '.pth')
        print('Loading completed !')

    def __getitem__(self, i):
        return self._volume[i], self._mask[i]

    def __len__(self):
        return self._volume.shape[0]

class Liverset3D(Dataset):
    def __init__(self, path, 
                    buffer=None, buffer_name="", len_dataset=-1, start_index=0, 
                    transforms=None, loader_fnc=Helper.loadTensor):
        super().__init__()
        self.path = path
        self.loader = loader_fnc
        self.buffer = buffer
        self.buffer_name = buffer_name
        self.start_index = int(start_index)
        _len = int(self.loader(path + '/info.pth').item())
        if len_dataset == -1:
            self.len = _len
        elif (len_dataset+start_index) > _len:
            self.len = _len - start_index
        else:
            self.len = len_dataset

    def __getitem__(self, i):
        if self.buffer is None:
            _vol = self.loader(self.path + '/volumes/vol_' + str(i) + '.pth')
            _mask = self.loader(self.path + '/masks/vol_' + str(i) + '.pth')
        else:
            i = i + self.start_index
            folder_index = i // self.buffer
            idx = i % self.buffer
            _vol  = self.loader(self.path + f'/volumes/{self.buffer_name}{str(folder_index)}/vol_{str(idx)}.pth')
            _mask = self.loader(self.path + f'/masks/{self.buffer_name}{str(folder_index)}/vol_{str(idx)}.pth')
        return _vol, _mask

    def __len__(self):
        return self.len

class Bloodset3D(Dataset):
    def __init__(self, path, 
                    buffer=None, buffer_name="", len_dataset=-1, start_index=0, 
                    transforms=None, loader_fnc=Helper.loadTensor):
        super().__init__()
        self.path = path
        self.loader = loader_fnc
        self.buffer = buffer
        self.buffer_name = buffer_name
        self.start_index = int(start_index)
        _len = int(self.loader(path + '/info.pth').item())
        if len_dataset == -1:
            self.len = _len
        elif (len_dataset+start_index) > _len:
            self.len = _len - start_index
        else:
            self.len = len_dataset
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        if self.buffer is None:
            _vol = self.loader(self.path + '/volumes/vol_' + str(i) + '.pth')
            _mask = self.loader(self.path + '/masks/vol_' + str(i) + '.pth')
        else:
            i = i + self.start_index
            folder_index = i // self.buffer
            _vol  = self.loader(self.path + f'/volumes/{self.buffer_name}{str(folder_index)}/vol_{str(i)}.pth')
            _mask = self.loader(self.path + f'/masks/{self.buffer_name}{str(folder_index)}/vol_{str(i)}.pth')
        return _vol, _mask

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import Dataset2D, Liverset3D, Bloodset3D
import os
from pathlib import Path

class Vessel3DLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, buffer=None, buffer_name="", len_dataset=-1, start_index=0, 
                    shuffle=False, validation_split=0.0, num_workers=6):
        _data_dir = str(Path(os.getcwd()).parent) + data_dir
        self.dataset = Bloodset3D(_data_dir, buffer, buffer_name, len_dataset, start_index)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class Liver3DLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, buffer=None, buffer_name="", len_dataset=-1, start_index=0, 
                    shuffle=False, validation_split=0.0, num_workers=6):
        _data_dir = str(Path(os.getcwd()).parent) + data_dir
        self.dataset = Liverset3D(_data_dir, buffer, buffer_name, len_dataset, start_index)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Data2DLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, eof_name, 
                    shuffle=False, validation_split=0.0, num_workers=6):
        _data_dir = str(Path(os.getcwd()).parent) + data_dir
        self.dataset = Dataset2D(_data_dir, eof_name)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

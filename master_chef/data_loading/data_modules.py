# data_modules.py
# Base imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def import_master_chef():
    try: 
        import master_chef
    except:
        exec(open('__init__.py').read()) 
        import master_chef
import_master_chef()

# Other imports
import torch.utils.data 
from torch.utils.data import dataset 
import torchvision
import os

# Class imports
from torch.utils.data.dataset import Dataset
from torch import Tensor
from master_chef.data_loading.toy_datasets import TabularDataset

# Master Chef imports
from master_chef.data_loading import toy_datasets


class ToyMNISTDM(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        toy_mnist: TabularDataset = toy_datasets.SklearnToys.mnist()
        n_samples = toy_mnist.n_samples
        train_size = round(0.9 * n_samples)
        self.train_val_size = train_size
        test_size = n_samples - train_size
        self.train_data, self.test_data = dataset.random_split(
            dataset=toy_mnist, lengths=[train_size, test_size])

    def setup(self, stage=None):
        if stage in ['fit', None]:
            train_size = round(0.9 * self.train_size)
            val_size = self.train_val_size - train_size

            self.train_data, self.val_data = dataset.random_split(
                dataset=self.train_data, 
                lengths = [train_size, val_size])

        if stage in ['test', None]:
            assert self.test_data is not None
        
    def get_dataloader(self, set: str = None):
        if set == "train":
            dl = torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batch_size)
        elif set == "val":
            dl = torch.utils.data.DataLoader(
                self.val_data, batch_size=self.batch_size)
        elif set == "test":
            dl = torch.utils.data.DataLoader(
                self.test_data, batch_size=self.batch_size)
        else:
            raise ValueError() # TODO: Write error message.
        return dl

    def train_dataloader(self):
        return self.get_dataloader(set='train')

    def val_dataloader(self):
        return self.get_dataloader(set='val')
        
    def test_dataloader(self):
        return self.get_dataloader(set='test')

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = os.path.join(os.getcwd(), 'data'),
                 batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size

    def prepare_data(self):
        data_dir = self.data_dir
        torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        data_dir = self.data_dir
        transform = torchvision.transforms.ToTensor()

        if stage in ['fit', None]:
            train_data = torchvision.datasets.MNIST(
                root=data_dir, train=True, transform=transform)
            self.train_data, self.val_data = dataset.random_split(
                dataset=train_data, lengths = [54000, 6000])

        if stage in ['test', None]:
            test_data = torchvision.datasets.MNIST(
                root=data_dir, train=False, transform=transform)
            self.test_data = test_data 
        
    def get_dataloader(self, set: str = None):
        if set == "train":
            dl = torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batch_size)
        elif set == "val":
            dl = torch.utils.data.DataLoader(
                self.val_data, batch_size=self.batch_size)
        elif set == "test":
            dl = torch.utils.data.DataLoader(
                self.test_data, batch_size=self.batch_size)
        else:
            raise ValueError() # TODO: Write error message.
        return dl

    def train_dataloader(self):
        return self.get_dataloader(set='train')

    def val_dataloader(self):
        return self.get_dataloader(set='val')
        
    def test_dataloader(self):
        return self.get_dataloader(set='test')

import os, sys
import torch
import numpy as np
from torch.utils.data import dataset
from numpy import ndarray
from torch import Tensor
from typing import Union
import sklearn.datasets

class TabularDataset(dataset.Dataset):
    def __init__(self, 
                 X: Union[ndarray, Tensor], 
                 Y: Union[ndarray, Tensor]):
        self.X = X
        self.Y = Y
        self.check_for_valid_inputs()
        self.convert_data_to_tensors()
        self.n_samples = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def check_for_valid_inputs(self):
        X, Y = self.X, self.Y
        assert X.ndim in [0, 1, 2], (
            f"The array dimension of X is too high. X.ndim: {X.ndim}")
        assert Y.ndim in [0, 1, 2], (
            f"The array dimension of Y is too high. Y.ndim: {Y.ndim}")
        assert X.shape[0] == Y.shape[0], (
            f"X and Y have different numbers of samples. Dim 0 should match.")
        assert isinstance(X, (ndarray, Tensor))
        assert isinstance(Y, (ndarray, Tensor))
    
    def convert_data_to_tensors(self):
        X, Y = self.X, self. Y
        
        if isinstance(X, ndarray):
            self.X = torch.from_numpy(X).float()
        elif isinstance(X, Tensor):
            self.X = X.float()
        else:
            raise Exception("Impossible!")

        if isinstance(Y, ndarray):
            Y = Y.reshape(-1)
            self.Y = torch.from_numpy(Y).long()
        elif isinstance(Y, Tensor):
            Y = Y.view(-1)
            self.Y = Y.long()
        else:
            raise Exception("Impossible!")
        
        assert isinstance(X, (ndarray, Tensor))

class SklearnToys:
    @staticmethod
    def mnist() -> dataset.Dataset: 
        sklearn_toy_ds = sklearn.datasets.load_digits()
        X = sklearn_toy_ds.data
        Y = sklearn_toy_ds.target
        sklearn_toy_ds = TabularDataset(X=X, Y=Y)
        return sklearn_toy_ds
    
    @staticmethod
    def diabetes() -> dataset.Dataset:
        sklearn_toy_ds = sklearn.datasets.load_diabetes()
        X = sklearn_toy_ds.data
        Y = sklearn_toy_ds.target
        sklearn_toy_ds = TabularDataset(X=X, Y=Y)
        return sklearn_toy_ds
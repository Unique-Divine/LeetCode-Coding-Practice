import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch import Tensor

class LitClassifier(pl.LightningModule):
    """PyTorch Lightning Module for supervised classification. """
    def __init__(self, 
                 model: nn.Module,
                 loss_fn,
                 lr: float,):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_function = loss_fn

        accuracy = pl.metrics.Accuracy()
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr = self.lr)
        return optimizer

    # --------------- Training and validation steps --------------- #

    def training_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        
        # Log step
        preds = torch.softmax(input=logits, dim=1)
        self.train_accuracy(preds=preds, target=y)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False,
                 prog_bar=False)
        self.log('train_acc_step', self.train_accuracy, on_step=True, 
                 on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        
        # Log step
        preds = torch.softmax(input=logits, dim=1)
        self.val_accuracy(preds=preds, target=y)
        self.log('val_loss_step', loss, on_step=True, on_epoch=False, 
                 prog_bar=True)
        self.log('val_acc_step', self.val_accuracy, on_step=True, 
                 on_epoch=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)

        # Log step
        preds = torch.softmax(input=logits, dim=1)
        self.test_accuracy(preds=preds, target=y)
        self.log('test_loss_step', loss, 
                 on_step=True, on_epoch=False)
        self.log('test_acc_step', self.test_accuracy, 
                 on_step=True, on_epoch=True)
        return loss

class LitRegressor(pl.LightningModule):
    """PyTorch Lightning Module for supervised regression."""
    def __init__(self, 
                 model: nn.Module,
                 loss_fn,
                 lr: float,):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_function = loss_fn

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr = self.lr)
        return optimizer

    # --------------- Training and validation steps --------------- #

    def training_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        preds = self(x)
        loss = self.loss_function(preds, y)
        
        # Log step
        self.log('train_loss_step', loss, on_step=True, on_epoch=False,
                 prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        preds = self(x)
        loss = self.loss_function(preds, y)
        
        # Log step
        self.log('val_loss_step', loss, on_step=True, on_epoch=False, 
                 prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Perform step
        x, y = batch
        preds = self(x)
        loss = self.loss_function(preds, y)

        # Log step
        self.log('test_loss_step', loss, 
                 on_step=True, on_epoch=False)
        return loss

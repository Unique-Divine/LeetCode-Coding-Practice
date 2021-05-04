import os, sys
import torch 
import torch.nn as nn
import pytorch_lightning as pl
def import_master_chef():
    try: 
        import master_chef
    except:
        exec(open('__init__.py').read()) 
        import master_chef
import_master_chef()
from master_chef import ffnn, lit_modules
from master_chef.data_loading import data_modules
import sklearn.datasets
import warnings
warnings.filterwarnings('ignore')


class TestModels:
    lr = 1e-3

    def test_classifier_quick_pass(self):
        data_module: pl.LightningDataModule = data_modules.ToyClassificationDM(
            batch_size = 50)
        
        img_dim = sklearn.datasets.load_digits().data.shape[1]

        network = ffnn.FFNNClassifier(
            input_dim = img_dim,
            num_classes = 10,
            num_hidden_layers = 1,)
        lit_module = lit_modules.LitClassifier(
            model = network, 
            loss_fn = nn.CrossEntropyLoss(), 
            lr = self.lr)

        trainer = pl.Trainer(gpus=0, fast_dev_run=True)
        trainer.fit(lit_module, datamodule=data_module)

    def test_regressor_quick_pass(self):
        data_module: pl.LightningDataModule = data_modules.ToyRegressionDM(
            batch_size = 50)
        
        input_dim = sklearn.datasets.load_diabetes().data.shape[1]

        network = ffnn.FFNNRegressor(
            input_dim = input_dim,
            num_hidden_layers = 1,)
        lit_module = lit_modules.LitRegressor(
            model = network, 
            loss_fn = nn.MSELoss(), 
            lr = self.lr)

        trainer = pl.Trainer(gpus=0, fast_dev_run=True)
        trainer.fit(lit_module, datamodule=data_module)

    # TODO: Write test for full MNIST?
        # Below code is used in full MNIST loading protocol.
        # current_file_parent_dir = os.path.dirname(os.path.realpath(__file__))
        # root_dir = os.path.dirname(current_file_parent_dir)
        # os.chdir(root_dir)

def manual_run():
    tm = TestModels()
    for test in [
        tm.test_classifier_quick_pass, 
        tm.test_regressor_quick_pass,]:
        test()
        print(f"Test '{test.__name__}' passed.")

# manual_run()

"""
Hide pytest warnings: https://stackoverflow.com/a/50821160/13305627

pytest -p no:warnings
"""
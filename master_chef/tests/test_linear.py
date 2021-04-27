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
from master_chef import linear
from master_chef import data_modules

import warnings
warnings.filterwarnings('ignore')


class TestModels:
    lr = 1e-3

    def test_quick_pass(self):

        current_file_parent_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(current_file_parent_dir)
        os.chdir(root_dir)
        data_module = data_modules.MNISTDataModule(
            data_dir=os.path.join(os.getcwd(), 'data'), 
            batch_size = 50
        )
        mnist_img_dims = (1, 28, 28)
        channels, width, height = mnist_img_dims  

        network = linear.FFNNClassifier(
            input_dim = channels * width * height,
            num_classes = 10,
            num_hidden_layers = 1,)
        lit_module = linear.LitClassifier(
            model = network, 
            loss_fn = nn.CrossEntropyLoss(), 
            lr = self.lr)

        trainer = pl.Trainer(gpus=0, fast_dev_run=True)
        trainer.fit(lit_module, datamodule=data_module)

def manual_run():
    for test in [\
        TestModels().test_quick_pass]:
        test()
        print(f"Test '{test.__name__}' passed.")

# manual_run()

"""
Hide pytest warnings: https://stackoverflow.com/a/50821160/13305627

pytest -p no:warnings
"""
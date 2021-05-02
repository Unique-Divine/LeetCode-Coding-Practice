import os, sys
def import_master_chef():
    global os, sys
    import os, sys
    try: 
        import master_chef
    except:
        exec(open('__init__.py').read()) 
        import master_chef
import_master_chef()
from master_chef import ffnn
from master_chef.data_loading import toy_datasets, data_modules 
from torch.utils.data import dataset
from torch import Tensor

class TestToyDatasets:
    def tabular_toy_dataset(self, kind: str = 'c'):
        if not isinstance(kind, str):
            raise ValueError(f"'kind' must be a string, not type {type(kind)}")
        if kind in ['classification', 'c']:
            kind = 'mnist'
        elif kind in ['regression', 'r']:
            kind = 'diabetes'        
        else:
            raise ValueError(
                "'kind' must be regression or classification, i.e. 'r' or 'c'")
        get_toy_ds = dict(
            mnist = toy_datasets.SklearnToys.mnist,
            diabetes = toy_datasets.SklearnToys.diabetes,
        )
        
        toy_dataset = get_toy_ds[kind]()
        assert isinstance(toy_dataset, dataset.Dataset)
        for x, y in toy_dataset:
            batch = x, y
            assert isinstance(x, Tensor)
            assert isinstance(y, Tensor)
            break

    def test_tabular_toy_datasets(self):
        self.tabular_toy_dataset(kind='c')
        self.tabular_toy_dataset(kind='r')


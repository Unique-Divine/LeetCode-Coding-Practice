def import_master_chef():
    global os, sys
    import os, sys
    try: 
        import master_chef
    except:
        exec(open('__init__.py').read()) 
        import master_chef
import_master_chef()
from master_chef import linear
from master_chef import data_modules

class TestImports:
    def test_root(self):
        current_file_parent_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(current_file_parent_dir)
        os.chdir(root_dir)
        exec(open('__init__.py').read()) 


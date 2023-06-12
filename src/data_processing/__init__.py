import scipy.io
import numpy as np
from ase import Atoms
from src.data_processing.molecule import get_molecule_name 
from src.data_processing.dataset.qm7 import  QM7DataML

def make_data_loader(cfg):
    if cfg['DATASET']['DATASET_NAME'] == 'qm7':
        data = QM7DataML(cfg['DATASET']['PATH'], cfg=cfg['DATASET'])
        data.process()
        train_loader, val_loader = data.data_train, data.data_test
        print("Loaderd QM7 dataset")
        # return train_loader, val_loader
    else:
        print("Dataset not found")
        exit(1)
    
    return train_loader, val_loader



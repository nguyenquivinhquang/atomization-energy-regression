import scipy.io
import numpy as np
from ase import Atoms
from src.data_processing.molecule import get_molecule_name
from src.data_processing.dataset.qm7 import QM7DataML, QM7DataGraph
from torch.utils.data import DataLoader
import torch

def make_data_loader_ml(cfg):
    if cfg["DATASET"]["DATASET_NAME"] == "qm7":
        data = QM7DataML(cfg["DATASET"]["PATH"], cfg=cfg["DATASET"])
        data.process()
        train_loader, val_loader = data.data_train, data.data_test
        print("Loaded QM7 dataset")
        scale_factor = data.scale_factor
        # return train_loader, val_loader
    else:
        print("Dataset not found")
        exit(1)

    return train_loader, val_loader, scale_factor

def make_data_loader(cfg):
    if cfg["DATASET"]["DATASET_NAME"] == "qm7":
        if cfg["DATASET"]["GRAPH_FEATURE"] == False:
            data = QM7DataML(cfg["DATASET"]["PATH"], cfg=cfg["DATASET"])
        else:
            data = QM7DataGraph(cfg["DATASET"]["PATH"], cfg=cfg["DATASET"])
        data.process()
        data_train, data_val = data.data_train, data.data_test
        print("Loaded QM7 dataset")
        scale_factor = data.scale_factor
        train_loader = {}
        val_loader = {}
        feature_size = data_train[0]['X'].shape[1]
        for fold in data_train:
            train = data_train[fold]
            train['X'] = torch.from_numpy(train['X']).float()
            train['Y'] = torch.from_numpy(train['Y']).float()
            train['Y'] = train['Y'].view(-1, 1)
            train_loader[fold] = DataLoader(dataset=list(zip(train['X'], train['Y'])), batch_size=cfg['SOLVER']["BATCH_SIZE"], shuffle=True)
        for fold in data_val:
            val = data_val[fold]
            val['X'] = torch.from_numpy(val['X']).float()
            val['Y'] = torch.from_numpy(val['Y']).float()
            val['Y'] = val['Y'].view(-1, 1)
            val_loader[fold] = DataLoader(dataset=list(zip(val['X'], val['Y'])), batch_size=cfg['SOLVER']["BATCH_SIZE"], shuffle=False)           
    else:
        print("Dataset not found")
        exit(1)

    return train_loader, val_loader, scale_factor, feature_size
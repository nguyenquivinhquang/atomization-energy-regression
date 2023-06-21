import scipy.io
import numpy as np
from ase import Atoms
from src.data_processing.molecule import get_molecule_name
from src.data_processing.dataset.qm7 import QM7DataML, QM7DataGraph
from torch.utils.data import DataLoader
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader


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
            data.process()
            data_train, data_val = data.data_train, data.data_test
            print("Loaded QM7 dataset")
            scale_factor = data.scale_factor
            train_loader = {}
            val_loader = {}
            feature_size = data_train[0]["X"].shape[1]
            for fold in data_train:
                train = data_train[fold]
                train_loader[fold] = DataLoader(
                    dataset=list(zip(train["X"], train["Y"])),
                    batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                    shuffle=True,
                )
            for fold in data_val:
                val = data_val[fold]
                val_loader[fold] = DataLoader(
                    dataset=list(zip(val["X"], val["Y"])),
                    batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                    shuffle=False,
                )
        else:
            data = QM7DataGraph(cfg["DATASET"]["PATH"], cfg=cfg["DATASET"])
            data.process()
            scale_factor = data.scale_factor
            train_loader = {}
            val_loader = {}
            for fold in data.data_train:
                train = data.data_train[fold]
                train_loader[fold] = GraphDataLoader(
                    train,
                    batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                    shuffle=True,
                )
            for fold in data.data_test:
                val = data.data_test[fold]
                val_loader[fold] = GraphDataLoader(
                    val,
                    batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                    # batch_size=1,
                    shuffle=False,
                )
            feature_size = train_loader[0].dataset[0].num_features
    else:
        print("Dataset not found")
        exit(1)

    return train_loader, val_loader, scale_factor, feature_size


# ./pdf_file/

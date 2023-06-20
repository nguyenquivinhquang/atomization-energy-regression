from .build_ml import build_ml_model
import os
from .GraphAttentionNetwork import GAT
from .graphConv import GCN
from src.model.mlp import MLP


def build_regression_model(cfg, input_size=0):
    if cfg["MODEL"]["MODEL_NAME"] == "MLP":
        model = MLP(input_size, cfg["MODEL"]["HIDDEN_LAYERS"])
    elif cfg["MODEL"]["MODEL_NAME"] == "GAT":
        model = GAT(
            dim_in=input_size,
            dim_h=cfg["MODEL"]["DIM_H"],
            heads=cfg["MODEL"]["HEADS"],
            edge_dim=cfg["MODEL"]["EDGE_DIM"],
        )
    elif cfg["MODEL"]["MODEL_NAME"] == "GCN":
        model = GCN(num_node_features=input_size, hidden_channels=cfg["MODEL"]["DIM_H"])
    else:
        model = build_ml_model(cfg["MODEL"])

    return model

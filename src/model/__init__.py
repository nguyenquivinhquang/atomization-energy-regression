from .build_ml import build_ml_model
import os
from .GraphAttentionNetwork import GAT
from .graphConv import GCN
from src.model.mlp import MLP
def build_regression_model(cfg, input_size = 0):
    if  cfg["MODEL"]["MODEL_NAME"] == 'MLP':
        model = MLP(input_size, cfg["MODEL"]["HIDDEN_LAYERS"])
    elif "GNN" not in cfg["MODEL"]["MODEL_NAME"]:
        model = build_ml_model(cfg["MODEL"])
    else:
        # build GNN model
        cfg_model = cfg["MODEL"]
        if cfg_model["MODEL_NAME"] == "GAT":
            model = GAT(dim_in=input_size,
                        dim_h=cfg_model["DIM_H"],
                        heads=cfg_model["HEADS"],
                        edge_dim=cfg_model["EDGE_DIM"])
        elif cfg_model["MODEL_NAME"] == "GCN":
            model = GCN(num_node_features=input_size,
                        hidden_channels=cfg_model["DIM_H"])
        
    return model

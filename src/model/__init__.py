from .build_ml import build_ml_model
import os
from src.model.mlp import MLP
def build_regression_model(cfg, input_size = 0):
    if  cfg["MODEL"]["MODEL_NAME"] == 'MLP':
        model = MLP(input_size, cfg["MODEL"]["HIDDEN_LAYERS"])
    elif "GNN" not in cfg["MODEL"]["MODEL_NAME"]:
        model = build_ml_model(cfg["MODEL"])
    else:
        pass
    return model

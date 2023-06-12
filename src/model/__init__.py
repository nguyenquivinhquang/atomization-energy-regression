from .build_ml import build_ml_model
import os

def build_regression_model(cfg):
    if 'GNN' not in cfg['MODEL']['MODEL_NAME']:
        model = build_ml_model(cfg['MODEL'])
    else:
        pass
    return model
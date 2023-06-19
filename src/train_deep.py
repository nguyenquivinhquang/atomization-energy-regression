from src.utils.opt import Opts, Config
from src.utils.utils import set_seed
from src.data_processing import make_data_loader
from src.model import build_regression_model
from processor.trainer.deep_trainer import Trainer
from processor.trainer import build_trainer
from sklearn.metrics import mean_absolute_error
from src.solver import make_optimizer, make_scheduler
import torch.nn as nn
import numpy as np
import torch

def train(cfg):
    print(cfg)
    train_loader, val_loader, scale, feature_size = make_data_loader(cfg)
    
    
    losses = []
    for idx in range(len(train_loader)):
        model = build_regression_model(cfg, feature_size)
        if cfg['SOLVER']['LOSS'] == 'MAE':
            loss_fn = nn.L1Loss()
        elif cfg['SOLVER']['LOSS'] == 'MSE':
            loss_fn = nn.MSELoss()
        optimizer = make_optimizer(cfg['SOLVER'], model)
        scheduler = make_scheduler(cfg['SOLVER'], optimizer)
        # trainer = Trainer(
        #     model, train_loader[idx], val_loader[idx], optimizer, scheduler, loss_fn, cfg["SOLVER"]["EPOCHS"]
        # )
        trainer = build_trainer(cfg, model, train_loader[idx], val_loader[idx], optimizer, scheduler, loss_fn, cfg["SOLVER"]["EPOCHS"])
        best_loss = trainer.process() * scale
        print("Best Loss: ", best_loss)
        losses.append(best_loss)
    losses = torch.tensor(losses)
    print("Average Loss: ", torch.mean(losses))
    return
if __name__ == "__main__":
    cfg = Opts(Config("configs/default.yaml")).parse_args()
    set_seed(cfg["SEED"])
    train(cfg)

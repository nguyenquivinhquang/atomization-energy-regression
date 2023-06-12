from src.utils.opt import Opts, Config
from src.utils.utils import set_seed
from src.data_processing import make_data_loader
from src.model import build_regression_model
from processor.trainer.ml_trainer import MLTrain
from sklearn.metrics import mean_absolute_error
# from src.
def train(cfg):
    print(cfg)
    train_loader, val_loader = make_data_loader(cfg)
    model = build_regression_model(cfg)

    trainer = MLTrain(model, train_loader, val_loader, mean_absolute_error, cfg)
    trainer.process()
    return

if __name__ == '__main__':
    cfg = Opts(Config("configs/default.yaml")).parse_args()
    set_seed(cfg["SEED"])
    train(cfg)

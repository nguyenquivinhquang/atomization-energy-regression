from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def build_ml_model(cfg):
    if cfg["MODEL_NAME"] == "KernelRidge":
        model = KernelRidge(
            kernel=cfg["KERNEL"], alpha=cfg["ALPHA"], gamma=cfg["GAMMA"]
        )
    elif cfg["MODEL_NAME"] == "SVR":
        model = SVR(kernel=cfg["KERNEL"], epsilon=cfg["EPSILON"], gamma=cfg["GAMMA"])
    elif cfg["MODEL_NAME"] == "LinearRegression":
        model = LinearRegression()
    elif cfg["MODEL_NAME"] == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(criterion="mae")
    # else:

    return model

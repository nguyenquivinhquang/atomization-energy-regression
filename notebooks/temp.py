import scipy.io
import numpy as np


def train_test_split(X, y, split_idx):
    mask = np.zeros(y.size, dtype=bool)
    mask[split_idx] = True
    X_train = X[~mask]
    y_train = y[~mask]
    X_test = X[mask]
    y_test = y[mask]
    return X_train, y_train, X_test, y_test

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def solve(X, y):
    Xhat = np.concatenate(
        (np.ones((X.shape[0], 1)), X),
        axis=1
    )    
    A = Xhat.T@Xhat
    b = Xhat.T@y
    w = np.linalg.pinv(A)@b

    return w

def predict(X, w):
    Xhat = np.concatenate(
        (np.ones((X.shape[0], 1)), X),
        axis=1
    )    
    return Xhat@w

def cross_val(X, y, split):
    score = []
    for sp in split:
        X_train, y_train, X_test, y_test = train_test_split(X, y, sp)
        w = solve(X_train, y_train)
        score.append(mae(predict(X_test, w), y_test))
    return np.mean(score)


if __name__ == "__main__":
    data = scipy.io.loadmat('dataset\qm7.mat')

    X = data['X']
    X_proc = []
    for x in X:
        X_proc.append(x[np.triu_indices(23)])
    X = np.vstack(X_proc)
    X = X.reshape(X.shape[0], -1)
    y = data['T'].T.squeeze()
    split = data['P']

    print(cross_val(X, y, split))
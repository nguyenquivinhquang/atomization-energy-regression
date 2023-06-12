from sklearn.metrics import mean_absolute_error
import numpy as np
from multiprocessing import Pool


class MLTrain(object):
    def __init__(self, model, train_loader, test_loader, metric, cfg, scale=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.metric = metric
        self.scale = 1
        if scale is not None:
            self.scale = scale
        print(self.model)

    def cross_validation(self):
        results = []
        for fold in self.train_loader:
            x_train, y_train = (
                self.train_loader[fold]["X"],
                self.train_loader[fold]["Y"],
            )
            x_test, y_test = self.test_loader[fold]["X"], self.test_loader[fold]["Y"]

            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            results.append(self.metric(y_pred, y_test))

        return np.asarray(results)

    def process(self):
        eval = self.cross_validation()
        folds_result = [
            "Fold " + str(idx) + ": " + str(i * self.scale) + ","
            for (idx, i) in enumerate(eval)
        ]
        print(folds_result)
        print("Average : ", np.mean(eval) * self.scale)
        print("Best:", np.min(eval) * self.scale)
        print("Worst:", np.max(eval) * self.scale)
        return (
            np.mean(eval) * self.scale,
            np.min(eval) * self.scale,
            np.max(eval) * self.scale,
        )

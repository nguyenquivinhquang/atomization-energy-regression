from sklearn.metrics import mean_absolute_error
import numpy as np
from multiprocessing import Pool

class MLTrain(object):
    def __init__(self, model, train_loader, test_loader, metric, cfg):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.metric = metric
    
    def cross_validation(self):
        results = []
        for fold in self.data_loader:
            x_train, y_train = self.data_loader[fold]['train']['X'], self.data_loader[fold]['train']['Y']
            x_test, y_test = self.data_loader[fold]['test']['X'], self.data_loader[fold]['test']['Y']
            
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            results.append(self.metric(y_pred, y_test))
        
        return np.asarray(results)
    
    def process(self):
        eval = self.cross_validation()
        folds_result = ["Fold: " + str(i) + ',' for i in eval]
        print(folds_result)
        print("Average : ", np.mean(eval))
        print("Best:", np.min(eval))
        print("Worst:", np.max(eval))
        return np.mean(eval), np.min(eval), np.max(eval)
import torch
import torch.nn as nn

import numpy as np


class Trainer(object):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        total_epochs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_epochs = total_epochs
        self.model.to(self.device)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
       
        with torch.no_grad():
            error = 0
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                MAE = nn.L1Loss(reduction='sum')
                error += MAE(output, target)
        val_loss /= len(self.val_loader.dataset)
        error /= len(self.val_loader.dataset)
        return val_loss, error

    def process(self):
        best_loss = np.inf
        for epoch in range(self.total_epochs):
            train_loss = self.train(epoch)
            val_loss, mae = self.validate(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            if epoch % 1 == 0:
                print(
                    "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t LR: {:.8f}".format(
                        epoch, train_loss, mae, current_lr
                    )
                )
            # print("Epoch: {}, MAE: {}".format(epoch, mae))
            if mae < best_loss:
                best_loss = mae
                # torch.save(self.model.state_dict(), 'model.pt')
                if epoch > 10:
                    print("Best loss at epoch {} is {:.6f}".format(epoch, best_loss))
        return best_loss


class GraphTrainer(Trainer):
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data) in enumerate(self.train_loader):
            data = data.to(self.device)
            output = self.model(
                data.x.float(), data.edge_index, data.batch, data.edge_attr
            )
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, data.y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            error = 0
            for data in self.val_loader:
                data = data.to(self.device)
                # output = self.model(data)
                output = self.model(data.x.float(), data.edge_index, data.batch, data.edge_attr)
   
                _val_loss = self.loss_fn(output, data.y).item()
                # _val_loss = self.loss_fn(output, gt).item()
                
                val_loss += _val_loss
                MAE = nn.L1Loss(reduction='sum')
                error += MAE(output, data.y)
               
        val_loss /= len(self.val_loader.dataset)
       
        error /= len(self.val_loader.dataset)
        return val_loss, error

from .deep_trainer import Trainer, DeepTrainer

def build_trainer(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs):
    if cfg['MODEL']['MODEL_NAME'] == 'MLP':
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    elif cfg['MODEL']['MODEL_NAME'] == 'GAT':
        trainer = DeepTrainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    elif cfg['MODEL']['MODEL_NAME'] == 'GCN':
        trainer = DeepTrainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    else:
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    return trainer
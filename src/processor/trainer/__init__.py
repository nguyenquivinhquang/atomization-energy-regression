from .deep_trainer import Trainer, GraphTrainer

def build_trainer(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs):
    if cfg['MODEL']['MODEL_NAME'] == 'MLP':
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    elif cfg['MODEL']['MODEL_NAME'] == 'GAT':
        trainer = GraphTrainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    elif cfg['MODEL']['MODEL_NAME'] == 'GCN':
        trainer = GraphTrainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    else:
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, total_epochs)
    return trainer
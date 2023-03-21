from torch import nn
from light_torch.module.base import Module
from light_torch.transform import Transform


class ClassificationModule(Module):
    def __init__(self, model: nn.Module, loss, transform: Transform = None):
        super().__init__()
        self.model = model
        self.transform = transform
        self.loss = loss

    def train_step(self, batch, batch_idx, epoch_idx):
        if self.transform is not None:
            batch = self.transform.fit(batch)

        xs, ys_gt = batch
        ys_pred = self.model(xs)
        loss_value = self.loss(ys_gt, ys_pred)
        return loss_value

    def predict(self, batch):
        if self.transform is not None:
            batch = self.transform.transform(batch)
        return self.model(batch)

    def val_step(self, batch, batch_idx, epoch_idx):
        if self.transform is not None:
            batch = self.transform.transform(batch)

        xs, ys_gt = batch
        ys_pred = self.model(xs)
        loss_value = self.loss(ys_gt, ys_pred)
        return loss_value

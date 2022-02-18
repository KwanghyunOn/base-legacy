import torch
from torch import nn

from .base import BaseTrainer


class ImageClsTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def train_fn(self):
        self.model.train()
        loss_sum = torch.tensor(0.0, device=self.device)
        num_data = 0
        for data in self.train_loader:
            imgs, labels = data["img"], data["label"]
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(imgs)
            loss = self.loss(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach() * len(imgs)
            num_data += len(imgs)

        return {"loss": loss_sum / num_data}

    @torch.no_grad()
    def eval_fn(self):
        self.model.eval()
        loss_sum = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        num_data = 0
        for data in self.eval_loader:
            imgs, labels = data["img"], data["label"]
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(imgs)
            loss = self.loss(logits, labels)
            predicted = torch.argmax(logits, dim=1)

            correct += (predicted == labels).sum()
            loss_sum += loss.detach() * len(imgs)
            num_data += len(imgs)

        return {"accuracy": correct / num_data, "loss": loss_sum / num_data}

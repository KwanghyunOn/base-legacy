import os

import torch
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        eval_loader,
        eval_every,
        update_ckpt_every,
        save_ckpt_every,
        exp_path,
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

        self.eval_every = eval_every
        self.update_ckpt_every = update_ckpt_every
        self.save_ckpt_every = save_ckpt_every

        self.exp_path = exp_path
        self.ckpt_path = os.path.join(self.exp_path, "ckpts")
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.writer = SummaryWriter(os.path.join(self.exp_path, "tb"))

        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train_fn(self, epoch):
        raise NotImplementedError

    def eval_fn(self, epoch):
        raise NotImplementedError

    def log_fn(self, epoch, train_dict, eval_dict):
        for metric_name, metric_value in train_dict.items():
            self.writer.add_scalar(
                f"train/{metric_name}", metric_value, global_step=epoch + 1
            )
        if eval_dict:
            for metric_name, metric_value in eval_dict.items():
                self.writer.add_scalar(
                    f"eval/{metric_name}", metric_value, global_step=epoch + 1
                )

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics) == 0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics) == 0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def save_checkpoint(self):
        ckpt = {
            "current_epoch": self.current_epoch,
            "train_metrics": self.train_metrics,
            "eval_metrics": self.eval_metrics,
            "eval_epochs": self.eval_epochs,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.current_epoch % self.update_ckpt_every == 0:
            torch.save(ckpt, os.path.join(self.ckpt_path, "latest.ckpt"))
        if self.current_epoch % self.save_ckpt_every == 0:
            torch.save(
                ckpt,
                os.path.join(self.ckpt_path, f"epoch{self.current_epoch:03d}.ckpt"),
            )

    def load_checkpoint(self):
        ckpt = torch.load(os.path.join(self.ckpt_path, "latest.ckpt"))
        self.current_epoch = ckpt["current_epoch"]
        self.train_metrics = ckpt["train_metrics"]
        self.eval_metrics = ckpt["eval_metrics"]
        self.eval_epochs = ckpt["eval_epochs"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def run(self, epochs, resume=False):
        if resume:
            self.load_checkpoint()
        elif os.path.isfile(os.path.join(self.ckpt_path, "latest.ckpt")):
            raise RuntimeError(
                "Checkpoint already exists. Remove the existing checkpoint or resume training by adding --resume argument"
            )

        for epoch in range(self.current_epoch, epochs):
            print(f"Epoch {epoch}")

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # Eval
            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
            else:
                eval_dict = None

            print("train", train_dict)
            print("eval", eval_dict)

            # Log
            self.log_fn(epoch, train_dict, eval_dict)

            # Checkpoint
            self.current_epoch += 1
            self.save_checkpoint()

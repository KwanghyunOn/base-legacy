import os
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import ddp


class BaseTrainer(metaclass=ABCMeta):
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        eval_loader,
        exp_path,
        eval_every,
        update_ckpt_every,
        save_ckpt_every,
        ddp=False,
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_every = eval_every
        self.update_ckpt_every = update_ckpt_every
        self.save_ckpt_every = save_ckpt_every

        self.exp_path = exp_path
        self.ckpt_path = os.path.join(self.exp_path, "ckpts")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.exp_path, "tb"))
        self.start_epoch = 0

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.ddp = ddp

        # For saving and loading state_dict
        if self.ddp:
            self.model_module = self.model.module
        else:
            self.model_module = self.model

    @abstractmethod
    def train_fn(self):
        """
        Return:
            Dictionary of training results. Every value in the dictionary must be a single-element tensor.
        """
        pass

    @abstractmethod
    def eval_fn(self):
        """
        Return:
            Dictionary of eval results. Every value in the dictionary must be a single-element tensor.
        """
        pass

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

    def save_checkpoint(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if (epoch + 1) % self.update_ckpt_every == 0:
            torch.save(ckpt, os.path.join(self.ckpt_path, "latest.ckpt"))
        if (epoch + 1) % self.save_ckpt_every == 0:
            torch.save(
                ckpt,
                os.path.join(self.ckpt_path, f"epoch{epoch:04d}.ckpt"),
            )

    def load_checkpoint(self):
        ckpt = torch.load(
            os.path.join(self.ckpt_path, "latest.ckpt"), map_location="cpu"
        )
        self.start_epoch = ckpt["epoch"] + 1
        self.model_module.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def run(self, epochs, resume=False):
        if resume:
            self.load_checkpoint()

        for epoch in range(self.start_epoch, epochs):
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)

            train_dict = self.train_fn()
            if self.ddp:
                train_dict = ddp.reduce_dict(train_dict)

            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn()
                if self.ddp:
                    eval_dict = ddp.reduce_dict(eval_dict)
            else:
                eval_dict = None

            if ddp.is_main_process():
                # Simple logging for ddp
                print(f"Epoch {epoch}")
                print(train_dict)
                if eval_dict:
                    print(eval_dict)
                # self.log_fn(epoch, train_dict, eval_dict)
                self.save_checkpoint(epoch)

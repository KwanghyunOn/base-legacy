import os
from abc import ABCMeta, abstractmethod

import torch

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
        use_ddp=False,
        logger=None,
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
        self.start_epoch = 0

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.use_ddp = use_ddp
        if self.use_ddp:
            self.model_module = self.model.module
        else:
            self.model_module = self.model
        self.is_master = ddp.is_master_process()
        self.logger = logger

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

    def save_checkpoint(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self._save_checkpoint_hook(ckpt)
        if (epoch + 1) % self.update_ckpt_every == 0:
            torch.save(ckpt, os.path.join(self.ckpt_path, "latest.ckpt"))
        if (epoch + 1) % self.save_ckpt_every == 0:
            torch.save(
                ckpt,
                os.path.join(self.ckpt_path, f"epoch{(epoch + 1):04d}.ckpt"),
            )

    def load_checkpoint(self):
        ckpt = torch.load(
            os.path.join(self.ckpt_path, "latest.ckpt"), map_location="cpu"
        )
        self.start_epoch = ckpt["epoch"] + 1
        self.model_module.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._load_checkpoint_hook(ckpt)

    def _save_checkpoint_hook(self, ckpt):
        """
        Override this function to add elements in checkpoint.
        """
        return

    def _load_checkpoint_hook(self, ckpt):
        """
        Override this function to load additional elements in checkpoint.
        """
        return

    def run(self, epochs, resume=False):
        if resume:
            self.load_checkpoint()

        for epoch in range(self.start_epoch, epochs):
            if self.use_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            self.logger.log(f"TRAINING Epoch {epoch}...", flush=True)
            train_dict = self.train_fn()
            self.logger.log(train_dict, channel="wandb", step=epoch)
            if (epoch + 1) % self.eval_every == 0:
                self.logger.log(f"EVALUATING Epoch {epoch}...", flush=True)
                eval_dict = self.eval_fn()
                self.logger.log(eval_dict, channel="wandb", step=epoch)

            if self.is_master:
                self.save_checkpoint(epoch)

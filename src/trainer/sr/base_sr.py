import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import ddp
from trainer.base import BaseTrainer
from utils.metrics import Evaluator


class SRTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = Evaluator(
            metrics=["psnr"], 
            data_range=2.0
        )
        self.scaler = torch.cuda.amp.GradScaler()
    
    def loss_fn(self, img_hr, img_sr):
        loss = F.l1_loss(img_hr, img_sr)
        return loss

    def train_fn(self):
        self.model.train()
        avg_loss = 0.0
        tq_train_loader = tqdm(self.train_loader, disable=(not self.is_master), ncols=80)
        for data in tq_train_loader:
            img_hr, img_lr = data["img_hr"], data["img_lr"]
            img_lr = img_lr.to(self.device)
            img_hr = img_hr.to(self.device)
            img_sr = self.model(img_lr)
            with torch.cuda.amp.autocast():
                loss = self.loss_fn(img_hr, img_sr)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tq_train_loader.set_description(f"[Loss: {loss.item():.4f}")
            avg_loss += loss.item()
        avg_loss /= len(self.train_loader)
        return {"loss": avg_loss}

    @torch.no_grad()
    def eval_fn(self):
        self.model.eval()
        results = {}
        tq_eval_loader = tqdm(self.eval_loader, disable=(not self.is_master), ncols=80)
        for data in tq_eval_loader:
            img_hr, img_lr = data["img_hr"], data["img_lr"]
            img_lr = img_lr.to(self.device)
            img_hr = img_hr.to(self.device)
            img_sr = self.model(img_lr)

            img_hr = ddp.gather_and_concat(img_hr)
            img_sr = ddp.gather_and_concat(img_sr)

            eval_result = self.evaluator(img_hr, img_sr)
            for k, v in eval_result.items():
                if k not in results:
                    results[k] = []
                results[k].append(v.item())

        for k in results:
            results[k] = sum(results[k]) / len(results[k])
        self.logger.log(results, flush=True)
        ddp.wait()
        return results
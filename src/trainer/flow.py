import torch

from .base import BaseTrainer
from utils import Measure


class FlowTrainer(BaseTrainer):
    def __init__(self, eval_metrics=["psnr"], **kwargs):
        super().__init__(**kwargs)
        self.measure = Measure(eval_metrics, device=self.device)

    def train_fn(self):
        self.model.train()
        loss_sum = 0.0
        num_iter = 0
        for data in self.train_loader:
            hr, lr = data["hr"], data["lr"]
            hr = hr.to(self.device)
            lr = lr.to(self.device)
            loss = self.model(hr=hr, lr=lr, sample=False)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach() * len(hr)
            num_iter += len(hr)
            bpd = loss_sum / num_iter
        return {"bpd": bpd}

    def eval_fn(self):
        self.model.eval()
        with torch.no_grad():
            eval_dict = {}
            eval_dict["bpd"] = 0.0
            num_iter = 0
            for data in self.eval_loader:
                hr, lr = data["hr"], data["lr"]
                batch_size = len(hr)
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                loss = self.model(hr, lr)
                eval_dict["bpd"] += loss.detach() * batch_size
                num_iter += batch_size

                hr_sample = self.model(lr=lr, sample=True)
                eval_result = self.measure(hr, hr_sample, data_range=2.0)
                for k, v in eval_result.items():
                    if k in eval_dict:
                        eval_dict[k] += v * batch_size
                    else:
                        eval_dict[k] = v * batch_size

            for k in eval_dict:
                eval_dict[k] = eval_dict[k] / num_iter
        return eval_dict

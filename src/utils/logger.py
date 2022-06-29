import wandb
import torch.distributed as dist


class Logger:
    def __init__(self, dir, use_wandb=False):
        self.disabled = dist.is_available() and \
                        dist.is_initialized() and \
                        dist.get_rank() != 0
        if self.disabled:
            return

        self.dir = dir
        if use_wandb:
            wandb.init(dir=self.dir, resume=True)
        self.use_wandb = use_wandb

    def log(self, *args, channel='stdout', **kwargs):
        if self.disabled:
            return
        if channel == 'stdout':
            self._log_stdout(*args, **kwargs)
        elif channel == 'wandb':
            if not self.use_wandb:
                return
            else:
                self._log_wandb(*args, **kwargs)
    
    def save_image(self, img, path):
        pass

    def _log_stdout(self, msg, **kwargs):
        print(msg, **kwargs) 
    
    def _log_wandb(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

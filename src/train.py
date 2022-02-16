import argparse

from model import get_model
from data import get_dataloader
from optim import get_optimizer
from trainer import get_trainer
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()
cfg = Config(args.config, args.resume, args.reset)

model = get_model(cfg.model_name, cfg.model_kwargs)
train_loader, eval_loader = get_dataloader(
    cfg.transform_name,
    cfg.transform_kwargs,
    cfg.dataset_name,
    cfg.dataset_kwargs,
    cfg.loader_kwargs,
)
optimizer = get_optimizer(cfg.optim_name, cfg.optim_kwargs, model.parameters())
trainer = get_trainer(
    cfg.trainer_name,
    cfg.trainer_kwargs,
    model,
    optimizer,
    train_loader,
    eval_loader,
)

trainer.run(cfg.epochs, args.resume)

import os
import argparse
import torch

from model import get_model
from data import get_dataloader, get_dataloader_ddp
from optim import get_optimizer
from trainer import get_trainer
from utils.logger import Logger
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    cfg = Config(args.config, args.resume, args.reset)
    if args.distributed:
        train_ddp(cfg)
    else:
        train(cfg)


def train(cfg):
    model = get_model(cfg.model_name, cfg.model_kwargs)
    loader_train, loader_eval = get_dataloader(
        cfg.transform_name,
        cfg.transform_kwargs,
        cfg.dataset_name,
        cfg.dataset_kwargs,
        cfg.loader_kwargs,
    )
    optimizer = get_optimizer(cfg.optim_name, cfg.optim_kwargs, model.parameters())
    logger = Logger(**cfg.logger_kwargs)
    trainer = get_trainer(
        cfg.trainer_name,
        cfg.trainer_kwargs,
        model,
        optimizer,
        loader_train,
        loader_eval,
        logger,
    )
    trainer.run(cfg.epochs, cfg.resume)


def train_ddp(cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model = get_model(cfg.model_name, cfg.model_kwargs).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    loader_train, loader_eval = get_dataloader_ddp(
        cfg.transform_name,
        cfg.transform_kwargs,
        cfg.dataset_name,
        cfg.dataset_kwargs,
        cfg.loader_kwargs,
    )
    optimizer = get_optimizer(cfg.optim_name, cfg.optim_kwargs, model.parameters())
    logger = Logger(**cfg.logger_kwargs)
    trainer = get_trainer(
        cfg.trainer_name,
        cfg.trainer_kwargs,
        model,
        optimizer,
        loader_train,
        loader_eval,
        logger,
        use_ddp=True,
    )
    trainer.run(cfg.epochs, cfg.resume)


if __name__ == "__main__":
    main()

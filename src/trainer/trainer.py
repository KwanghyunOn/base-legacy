from importlib import import_module


def get_trainer(
    trainer_name, trainer_kwargs, model, optimizer, train_loader, eval_loader, use_ddp=False
):
    module, attr = trainer_name.rsplit(".", 1)
    trainer_cls = getattr(import_module("." + module, "trainer"), attr)
    return trainer_cls(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        use_ddp=use_ddp,
        **trainer_kwargs
    )

import os
import shutil
import yaml


class Config:
    def __init__(self, cfg_path, resume, reset):
        self.cfg_path = cfg_path
        self.resume = resume
        self.reset = reset
        with open(self.cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self._parse_config()
        self._set_dirs()

    def _parse_config(self):
        self.model_name = self._get_value("model", "name", required=True)
        self.model_kwargs = self._get_value("model", "kwargs", default=dict())

        self.transform_name = self._get_value("data", "transform", "name")
        self.transform_kwargs = self._get_value(
            "data", "transform", "kwargs", default=dict()
        )

        self.dataset_name = self._get_value("data", "dataset", "name", required=True)
        self.dataset_kwargs = self._get_value(
            "data", "dataset", "kwargs", default=dict()
        )
        self.loader_kwargs = self._get_value("data", "loader", "kwargs", default=dict())

        self.optim_name = self._get_value("optim", "name", required=True)
        self.optim_kwargs = self._get_value("optim", "kwargs", default=dict())

        self.trainer_name = self._get_value("trainer", "name", required=True)
        self.trainer_kwargs = self._get_value("trainer", "kwargs", default=dict())
        self.logger_kwargs = self._get_value("logger", "kwargs", default=dict())

        self.epochs = self._get_value("train", "epochs", required=True)

    def _set_dirs(self):
        self.exp_path = os.path.join(
            self._get_value("exp", "root"),
            self._get_value("exp", "name"),
            self._get_value("exp", "ablation"),
        )
        ckpt_path = os.path.join(self.exp_path, "ckpts", "latest.ckpt")
        if self.reset:
            shutil.rmtree(self.exp_path, ignore_errors=True)
        elif os.path.exists(ckpt_path) and not self.resume:
            raise Exception(
                "Checkpoint found. Add --resume to resume the experiment or --reset to remove the existing training results."
            )
        os.makedirs(self.exp_path, exist_ok=True)
        self.trainer_kwargs["exp_path"] = self.exp_path
        log_dir = os.path.join(self.exp_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger_kwargs["dir"] = log_dir
        shutil.copyfile(self.cfg_path, os.path.join(self.exp_path, "config.yaml"))

    def _get_value(self, *keys, default=None, required=False):
        d = self.cfg
        for k in list(keys):
            if k not in d:
                if required:
                    raise Exception("Required keys not found in config: ", *keys)
                return default
            else:
                d = d[k]
        return d

import os
import shutil
import yaml


class Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        with open(self.cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self._parse_config()
        self._set_dirs()

    def _parse_config(self):
        self.model_name = self.cfg["model"]["name"]
        self.model_kwargs = self.cfg["model"]["kwargs"]

        self.dataset_name = self.cfg["data"]["dataset"]["name"]
        self.dataset_kwargs = self.cfg["data"]["dataset"]["kwargs"]
        self.loader_kwargs = self.cfg["data"]["loader"]["kwargs"]

        self.optim_name = self.cfg["optim"]["name"]
        self.optim_kwargs = self.cfg["optim"]["kwargs"]

        self.trainer_name = self.cfg["trainer"]["name"]
        self.trainer_kwargs = self.cfg["trainer"]["kwargs"]

        self.epochs = self.cfg["train"]["epochs"]

    def _set_dirs(self):
        self.exp_path = os.path.join(
            self.cfg["exp"]["root"],
            self.cfg["exp"]["name"],
            self.cfg["exp"]["ablation"],
        )
        os.makedirs(self.exp_path, exist_ok=True)
        self.trainer_kwargs["exp_path"] = self.exp_path
        shutil.copyfile(self.cfg_path, os.path.join(self.exp_path, "config.yaml"))

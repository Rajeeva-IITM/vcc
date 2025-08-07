# Small file to generate yaml file from Optuna logs

import optuna
from omegaconf import OmegaConf


def main(storage_db: str):
    study = optuna.create_study(storage=storage_db, load_if_exists=True)

    best_params = OmegaConf.create(study.best_params)

    return best_params  # TODO: Finish later

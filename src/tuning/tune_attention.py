from copy import deepcopy
from os import makedirs
from pathlib import Path
from pprint import pprint

import hydra
import optuna
import rootutils
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig
from optuna.integration import (
    PyTorchLightningPruningCallback,  # pyright: ignore[reportAttributeAccessIssue]
)
from rich.console import Console

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)
torch.cuda.empty_cache()

console = Console(record=True)
torch.set_float32_matmul_precision("high")


def generate_hidden_layers(
    trial: optuna.Trial, name: str, ascending: bool
) -> list[int]:
    # if input_size < output_size: # From small input, need to create large output
    #     ascending = False
    # else:
    #     ascending = True

    num_layers = trial.suggest_int(f"num_layers_{name}", 2, 8)
    hidden_size = trial.suggest_categorical(
        f"hidden_size_{name}", [256, 512, 1024, 2048, 4096, 8192]
    )
    hidden_layers: list[int] = []
    for _ in range(num_layers):
        hidden_layers.append(hidden_size)
        hidden_size = hidden_size // 2

    if ascending:
        hidden_layers.sort()

    return hidden_layers


def objective(trial: optuna.Trial, conf: DictConfig) -> float:
    torch.cuda.empty_cache()

    model_conf = deepcopy(conf.model)  # It modifies in place and causes errors

    # ic('Before', model_conf)

    embedding_size = trial.suggest_categorical("embedding_size", [128, 256, 512])
    activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "silu", "softplus"]
    )  # Need positive values

    # Hyperparameters one by one: KO processor

    processing_layers = {
        "ko_processor_args": "ko_processor",
        "exp_processor_args": "exp_processor",
        "decoder_args": "decoder",
    }

    for key in processing_layers:
        if key == "decoder_args":  # Decoder slightly different
            model_conf.net[key].input_size = embedding_size
            model_conf.net[key].hidden_layers = generate_hidden_layers(
                trial, processing_layers[key], True
            )
        else:  # Processors
            model_conf.net[key].output_size = embedding_size
            model_conf.net[key].hidden_layers = generate_hidden_layers(
                trial, processing_layers[key], False
            )
        # common for both decoder and processors
        model_conf.net[key].dropout = trial.suggest_float(
            f"dropout_{processing_layers[key]}", 0, 0.9
        )
        model_conf.net[key].activation = activation

    # Attention arguments

    model_conf.net.fusion_type = trial.suggest_categorical(
        "fusion_type", ["sum", "product", "cross_attn"]
    )
    model_conf.net.attention_args.embed_dim = embedding_size
    model_conf.net.attention_args.num_heads = 2 ** trial.suggest_int(
        "attention_heads", 0, 2
    )

    model_conf.optimizer.lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    model_conf.optimizer.weight_decay = trial.suggest_float(
        "learning_rate", 1e-6, 1e-2, log=True
    )

    model_conf.scheduler.max_lr = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True)

    # ic(model_conf)

    datamodule = instantiate(conf.data.datamodule)

    callbacks = [hydra.utils.instantiate(conf.callbacks[cb]) for cb in conf.callbacks]
    model = instantiate(model_conf)

    trainer: Trainer = instantiate(
        conf.trainer,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor=conf.optuna.objective),
            *callbacks,
        ],
    )
    trainer.fit(model, datamodule)

    return trainer.callback_metrics[conf.optuna.objective].item()
    #
    # return torch.randn(1).item()


@hydra.main(version_base=None, config_path="../../config", config_name="tune")
def main(conf: DictConfig):
    pprint(conf)

    groupname: str = conf.data.metadata.groupname
    savename: str = conf.data.metadata.savename

    # pbar_theme: RichProgressBarTheme = hydra.utils.instantiate(conf.callbacks.get("rich_progress_bar"))
    # pbar = RichProgressBar(theme=pbar_theme)

    file_path: Path = Path(conf.paths.log_dir) / savename

    console.print(f"[royal_blue1]Optuna Tuning - {groupname}[/royal_blue1]")
    save_file = file_path / "tuning.log"
    if not file_path.exists():
        makedirs(file_path, exist_ok=True)
        with open(save_file, "w") as _:
            console.print(f"File at [sandy_brown]{file_path}[/sandy_brown] created.")

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(  # pyright: ignore[reportAttributeAccessIssue]
            str(save_file)
        )
    )
    sampler = hydra.utils.instantiate(conf.get("optuna.sampler"))

    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        study_name=groupname,
        direction=conf.optuna.direction,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(
        lambda trial: objective(trial, conf),
        n_trials=conf.optuna.n_trials,
        gc_after_trial=True,
        n_jobs=conf.optuna.n_jobs,
    )

    console.print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial

    console.print(
        "Best trial:",
    )

    console.print(f"Value: {trial.value}")

    console.print("Params: ")
    console.print(trial.params)


if __name__ == "__main__":
    main()

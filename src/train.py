import hydra
import lightning
import rich
import rootutils
import torch
import wandb
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)
torch.cuda.empty_cache()

console = rich.console.Console()
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../config/", config_name="train.yaml")
def main(conf: DictConfig):
    """
    The main train file
    """
    if conf.get("seed"):
        lightning.seed_everything(conf.seed, workers=True)

    console.log(f"Instantiating datamodule: {conf.data.datamodule._target_}")

    datamodule: LightningDataModule = hydra.utils.instantiate(conf.data.datamodule)
    # ic(datamodule)

    console.log(f"Instantiating model: {conf.model._target_}")

    model: LightningModule = hydra.utils.instantiate(conf.model)
    if conf.compile:
        model.compile(mode="reduce-overhead")

    console.log("Instantiating callbacks")

    callbacks = [hydra.utils.instantiate(conf.callbacks[cb]) for cb in conf.callbacks]

    console.log(f"Instantiating Logger: {conf.logging.wandb._target_}")

    logger = hydra.utils.instantiate(conf.logging.wandb)
    logger.experiment.config.update(OmegaConf.to_container(conf))

    console.log(f"Instantiating Trainer: {conf.trainer._target_}")

    trainer: Trainer = hydra.utils.instantiate(
        conf.trainer, logger=logger, callbacks=[*callbacks]
    )

    trainer.fit(model, datamodule, ckpt_path=conf.get("ckpt_path"))
    wandb.finish()


if __name__ == "__main__":
    main()

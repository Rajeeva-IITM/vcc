import traceback

import hydra
import lightning
import rich
import rootutils
import torch
import wandb
from dotenv import load_dotenv
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)
from src.utils.umap_utilities import perform_umap, plot_output_plotly  # noqa: E402

# The configs resolve paths via ${oc.env:...}. Load .env before Hydra composes
# them so the file the README asks you to create is actually honoured, rather
# than requiring the variables to be exported by hand.
load_dotenv()

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

    logger: lightning.pytorch.loggers.WandbLogger = hydra.utils.instantiate(
        conf.logging.wandb
    )
    logger.experiment.config.update(OmegaConf.to_container(conf))

    console.log(f"Instantiating Trainer: {conf.trainer._target_}")

    trainer: Trainer = hydra.utils.instantiate(
        conf.trainer, logger=logger, callbacks=[*callbacks]
    )

    trainer.fit(model, datamodule, ckpt_path=conf.get("ckpt_path"))

    try:
        if not conf.trainer.fast_dev_run:
            save_path = conf.callbacks.model_checkpoint.dirpath
            console.log("Prediction and quick evaluation")
            preds: list[torch.Tensor] = trainer.predict(model, datamodule)
            if (
                len(preds[0]) == 2
            ):  # For the case where both final prediction and the projector info is given
                preds = [pred[0] for pred in preds]
            y_pred = torch.cat(preds)
            console.log(f"Predicted expression shape: {y_pred.shape}")
            torch.save(y_pred, save_path + "/predictions.pt")

            # Only the embedding datamodule exposes per-cell gene labels; skip
            # the plot rather than failing the run if they are unavailable.
            genes = getattr(datamodule.test_data, "perturbed_genes", None)
            if genes is None:
                console.log("test_data has no `perturbed_genes`; skipping UMAP")
            else:
                console.log("Running UMAP")
                # UMAP goes through numpy, which has no bfloat16; under
                # `bf16-mixed` the predictions come back as bfloat16.
                reduced = perform_umap(y_pred.float(), genes=genes)
                fig = plot_output_plotly(reduced)
                fig.write_html(save_path + "/UMAP-figure.html", auto_play=False)
                table = wandb.Table(columns=["UMAP-figure"])
                table.add_data(wandb.Html(save_path + "/UMAP-figure.html"))

            # logger.experiment.log({"UMAP-figure": wandb.Plotly(fig)})

    except Exception:
        # Log the full traceback: a bare `except` here previously hid a broken
        # predict path behind a one-line message for weeks.
        console.log("[red]Prediction/evaluation step failed:[/red]")
        console.log(traceback.format_exc())
    wandb.finish()


if __name__ == "__main__":
    main()

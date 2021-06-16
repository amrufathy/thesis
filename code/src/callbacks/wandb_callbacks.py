import glob
import os

import wandb
from datasets import load_from_disk
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class TextGenerationCallback(Callback):
    def __init__(self, data_dir: str, limit: int = 100):
        self.data_dir = data_dir
        self.limit = limit

    def on_fit_end(self, trainer, pl_module):
        dataset = load_from_disk(self.data_dir)["val"]  # use only val data
        tokenizer = pl_module.tokenizer

        gold_stories = tokenizer.batch_decode(
            dataset["story_ids"][: self.limit],
            skip_special_tokens=True,
        )
        gold_summaries = tokenizer.batch_decode(
            dataset["summary_ids"][: self.limit],
            skip_special_tokens=True,
        )

        del dataset

        if "Compressor" in str(pl_module.__class__):
            predictions = pl_module.generate(gold_stories)
        elif "Expander" in str(pl_module.__class__):
            predictions = pl_module.generate(gold_summaries)
        else:  # TODO: cycle model case
            predictions = []

        table = wandb.Table(columns=["gold_story", "gold_summary", "prediction"])

        for p, st, sm in zip(predictions, gold_stories, gold_summaries):
            table.add_data(st, sm, p)

        # fmt: off
        wandb.log({"generations": table})
        # fmt: on

        del predictions, gold_stories, gold_summaries

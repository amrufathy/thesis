from typing import Optional

import dotenv
import hydra
import nltk
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything

from src.models import CycleModel, ExpanderModel
from src.utils import utils
from src.utils.eval_metrics import bleu

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

nltk.download("punkt")

log = utils.get_logger(__name__)

LIMIT = 100


def compare_outputs(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init Lightning model
    log.info("Instantiating models")

    MODEL_NAME = "facebook/bart-base"

    # TODO: load from checkpoints
    model_exp = ExpanderModel(model_name_or_path=MODEL_NAME)
    model_cycle_exp = CycleModel(expander_model_name=MODEL_NAME, compressor_model_name=MODEL_NAME, direction="exp")
    model_cycle_comp = CycleModel(expander_model_name=MODEL_NAME, compressor_model_name=MODEL_NAME, direction="comp")

    # Test the model
    log.info("Starting testing!")

    df = pd.DataFrame(
        columns=["Gold story", "[P] Expander", "BLEU Exp", "[P] C-Exp", "BLEU C-Exp", "[P] C-Comp", "BLEU C-Comp"]
    )

    for i, batch in enumerate(datamodule.test_dataloader()):
        gold_summary = model_exp.arch.ids_to_clean_text(batch["summary_ids"])[0]
        gold_story = model_exp.arch.ids_to_clean_text(batch["story_ids"])[0]

        pred_story_exp = model_exp.arch.generate_from_text(gold_summary)[0]
        pred_story_c_exp = model_cycle_exp.arch.expander.generate_from_text(gold_summary)[0]
        pred_story_c_comp = model_cycle_comp.arch.expander.generate_from_text(gold_summary)[0]

        bleu_exp = bleu(gold_summary, pred_story_exp)
        bleu_c_exp = bleu(gold_summary, pred_story_c_exp)
        bleu_c_comp = bleu(gold_summary, pred_story_c_comp)

        df = df.append(
            {
                "Gold story": gold_story,
                "[P] Expander": pred_story_exp,
                "BLEU Exp": bleu_exp["bleu"],
                "[P] C-Exp": pred_story_c_exp,
                "BLEU C-Exp": bleu_c_exp["bleu"],
                "[P] C-Comp": pred_story_c_comp,
                "BLEU C-Comp": bleu_c_comp["bleu"],
            },
            ignore_index=True,
        )

        if i > LIMIT:
            break

    log.info("Writing csv file")
    df.to_csv('stories_comparison.csv', index=False)


@hydra.main(config_path="../../configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, fields=["datamodule"], resolve=True)

    return compare_outputs(config)


if __name__ == "__main__":
    main()

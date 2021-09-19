import sys  # isort:skip

# sys.path.append("/content/")

from typing import Optional

import dotenv
import hydra
import nltk
import pandas as pd
from datasets import load_metric
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything
from tqdm import tqdm

from src.models import CycleModel, ExpanderModel
from src.utils import utils
from src.utils.eval_metrics import bleu

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

nltk.download("punkt")

log = utils.get_logger(__name__)

LIMIT = 100

hf_bleurt = load_metric("bleurt")


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
        columns=[
            "Gold summary",
            "Gold story",
            "[P] Expander",
            "BLEU Exp",
            "BLEURT Exp",
            "[P] C-Exp",
            "BLEU C-Exp",
            "BLEURT C-Exp",
            "[P] C-Comp",
            "BLEU C-Comp",
            "BLEURT C-Comp",
        ]
    )

    # TODO: set batch size in config to appropriate size (~100)
    batch = next(iter(datamodule.test_dataloader()))

    gold_summaries = model_exp.arch.ids_to_clean_text(batch["summary_ids"])
    gold_stories = model_exp.arch.ids_to_clean_text(batch["story_ids"])

    pred_stories_exp = model_exp.arch.generate_from_text(gold_summaries)
    pred_stories_c_exp = model_cycle_exp.arch.expander.generate_from_text(gold_summaries)
    pred_stories_c_comp = model_cycle_comp.arch.expander.generate_from_text(gold_summaries)

    for i, (gold_summary, gold_story, pred_story_exp, pred_story_c_exp, pred_story_c_comp) in tqdm(
        enumerate(zip(gold_summaries, gold_stories, pred_stories_exp, pred_stories_c_exp, pred_stories_c_comp)),
        total=len(gold_summaries),
    ):

        bleu_exp = bleu([gold_story], [pred_story_exp])
        bleu_c_exp = bleu([gold_story], [pred_story_c_exp])
        bleu_c_comp = bleu([gold_story], [pred_story_c_comp])

        bleurt_exp = hf_bleurt.compute(references=[gold_story], predictions=[pred_story_exp])["scores"][0]
        bleurt_c_exp = hf_bleurt.compute(references=[gold_story], predictions=[pred_story_c_exp])["scores"][0]
        bleurt_c_comp = hf_bleurt.compute(references=[gold_story], predictions=[pred_story_c_comp])["scores"][0]

        df = df.append(
            {
                "Gold summary": gold_summary,
                "Gold story": gold_story,
                "[P] Expander": pred_story_exp,
                "BLEU Exp": bleu_exp["bleu2"],
                "BLEURT Exp": bleurt_exp,
                "[P] C-Exp": pred_story_c_exp,
                "BLEU C-Exp": bleu_c_exp["bleu2"],
                "BLEURT C-Exp": bleurt_c_exp,
                "[P] C-Comp": pred_story_c_comp,
                "BLEU C-Comp": bleu_c_comp["bleu2"],
                "BLEURT C-Comp": bleurt_c_comp,
            },
            ignore_index=True,
        )

        if i > LIMIT:
            break

    log.info("Writing csv file")
    df.to_csv("stories_comparison.csv", index=False)


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

from typing import Dict, List, Optional, Union

from pytorch_lightning import LightningModule
from torch import argmax
from transformers import BartTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import Compressor
from src.utils.eval_metrics import bleu


class CompressorModel(LightningModule):
    def __init__(self, model_name_or_path: str, learning_rate: float = 5e-5, max_generation_length: int = 7, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.arch = Compressor(model_name_or_path=model_name_or_path, max_generation_length=max_generation_length)
        self.lr = learning_rate

        self.references = []
        self.predictions = []

    def forward(self, dict_input: Dict) -> Dict:
        return self.arch(dict_input)

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log("train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        # fmt: on

        return results["loss"]

    def log_at_val_test_epoch_end(self, metrics: Dict, stage_prefix: str):
        for key in metrics.keys():
            self.log(f"{stage_prefix}/{key}", metrics[key], on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Calculate batch-level micro-bleu on model output logits
            as approximation for model selection
        """

        results = self.forward(batch)

        # fmt: off
        self.log("val/loss", results["loss"], on_step=False, on_epoch=True)
        # fmt: on

        logits = argmax(results["logits"], dim=-1)
        predictions = self.arch.ids_to_clean_text(logits)
        targets = self.arch.ids_to_clean_text(batch["summary_ids"])

        metrics = {
            "bleu": bleu(targets, predictions)["bleu"],
        }

        self.log_at_val_test_epoch_end(metrics, "val")

        return results["loss"]

    def test_step(self, batch, batch_idx):
        """
        Accumulate targets and model-generated predictions to calculate
            corpus-level metrics at epoch end
        """

        results = self.forward(batch)

        # fmt: off
        self.log("test/loss", results["loss"], on_step=False, on_epoch=True)
        # fmt: on

        # accumulate bleu sys & refs
        targets = self.arch.ids_to_clean_text(batch["summary_ids"])
        predictions = self.arch.generate_from_ids(
            {"input_ids": batch["story_ids"], "attention_mask": batch["story_attn_msk"]}
        )

        self.references.extend(targets)
        self.predictions.extend(predictions)

        return results["loss"]

    def on_test_epoch_end(self):
        """
        Calculate corpus-level metrics on targets and
            model-generated predictions
        """

        metrics = {"bleu": bleu(self.references, self.predictions)["bleu"]}

        self.log_at_val_test_epoch_end(metrics, "test")

        self.predictions, self.references = [], []

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> List[str]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.arch.generate_from_text(conditioning_sentences)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    @property
    def tokenizer(self) -> BartTokenizerFast:
        return self.arch.tokenizer

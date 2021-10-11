from typing import Dict, List, Union

import numpy as np
from datasets import load_metric
from pytorch_lightning import LightningModule
from torch import argmax, tensor
from transformers import BartTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import Expander
from src.utils.eval_metrics import bleu, distinct_n


class ExpanderModel(LightningModule):
    def __init__(self, model_name_or_path: str, learning_rate: float = 5e-5, max_generation_length: int = 70, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.arch = Expander(
            model_name_or_path=model_name_or_path,
            max_generation_length=max_generation_length,
        )
        self.lr = learning_rate

        self.bleurt = load_metric("bleurt", "bleurt-base-512")
        self.bert_score = load_metric("bertscore")

        self.references = []
        self.predictions = []

    def forward(self, dict_input: Dict) -> Dict:
        return self.arch(dict_input)

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log("train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", results["ppl"], on_step=False, on_epoch=True)
        # fmt: on

        return results["loss"]

    def log_at_val_test_epoch_end(self, metrics: Dict, stage_prefix: str):
        for key in metrics.keys():
            self.log(f"{stage_prefix}/{key}", metrics[key], on_step=False, on_epoch=True)

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Calculate batch-level micro-bleu on model output logits
            as approximation for model selection
        """

        results = self.forward(batch)

        # fmt: off
        self.log("val/loss", results["loss"], on_step=False, on_epoch=True)
        self.log("val/ppl", results["ppl"], on_step=False, on_epoch=True)
        # fmt: on

        logits = argmax(results["logits"], dim=-1)
        predictions = self.arch.ids_to_clean_text(logits)
        targets = self.arch.ids_to_clean_text(batch["story_ids"])

        metrics = {**bleu(targets, predictions), **distinct_n(predictions)}

        for k, v in metrics.items():
            metrics[k] = tensor(v, device=self.device)

        self.log_at_val_test_epoch_end(metrics, "val")

        return results["loss"]

    def test_step(self, batch: Dict, batch_idx: int):
        """
        Accumulate targets and model-generated predictions to calculate
            corpus-level metrics at epoch end
        """

        results = self.forward(batch)

        # fmt: off
        self.log("test/loss", results["loss"], on_step=False, on_epoch=True)
        self.log("test/ppl", results["ppl"], on_step=False, on_epoch=True)
        # fmt: on

        # accumulate bleu sys & refs
        targets = self.arch.ids_to_clean_text(batch["story_ids"])
        predictions = self.arch.generate_from_ids(
            {"input_ids": batch["summary_ids"], "attention_mask": batch["summary_attn_msk"]}
        )

        self.references.extend(targets)
        self.predictions.extend(predictions)

        self.bleurt.add_batch(predictions=predictions, references=targets)
        self.bert_score.add_batch(predictions=predictions, references=targets)

        return results["loss"]

    def on_test_epoch_end(self):
        """
        Calculate corpus-level metrics on targets and
            model-generated predictions
        """

        metrics = {
            **bleu(self.references, self.predictions),
            **distinct_n(self.predictions),
            "bleurt": np.mean(self.bleurt.compute()["scores"]),
            "bertscore": np.mean(self.bertscore.compute(lang="en", rescale_with_baseline=True)["f1"]),
        }

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

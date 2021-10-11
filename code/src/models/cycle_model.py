from typing import Dict, List, Tuple, Union

import numpy as np
from datasets import load_metric
from pytorch_lightning import LightningModule
from torch import argmax, tensor
from transformers import BartTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import (
    CycleArchitectureCompress,
    CycleArchitectureDual,
    CycleArchitectureExpand,
)
from src.utils.eval_metrics import bleu, distinct_n


class CycleModel(LightningModule):
    def __init__(
        self,
        expander_model_name: str,
        compressor_model_name: str,
        direction: str,
        use_gumbel_softmax: bool = False,
        expander_learning_rate: float = 5e-5,
        compressor_learning_rate: float = 5e-5,
        max_story_length: int = 70,
        max_summary_length: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        cycle_arch_params = {
            "expander_model_name": expander_model_name,
            "compressor_model_name": compressor_model_name,
            "use_gumbel_softmax": use_gumbel_softmax,
            "max_story_length": max_story_length,
            "max_summary_length": max_summary_length,
            **kwargs,
        }

        if direction in {"comp", "compress", "compressor"}:
            self.arch = CycleArchitectureCompress(**cycle_arch_params)
        elif direction in {"exp", "expand", "expander"}:
            self.arch = CycleArchitectureExpand(**cycle_arch_params)
        else:
            self.arch = CycleArchitectureDual(**cycle_arch_params)

        self.comp_lr = compressor_learning_rate
        self.exp_lr = expander_learning_rate

        self.bleurt = load_metric("bleurt", "bleurt-base-512")
        self.bert_score = load_metric("bertscore")

        self.story_references, self.summary_references = [], []
        self.story_predictions, self.summary_predictions = [], []

    def forward(self, dict_input: Dict) -> Dict:
        return self.arch(dict_input)

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log("train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/exp_loss", results["exp_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/comp_loss", results["comp_loss"], on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/sem_loss", results["sem_loss"], on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/exp_ppl", results["exp_ppl"], on_step=False, on_epoch=True)
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
        self.log("val/exp_loss", results["exp_loss"], on_step=False, on_epoch=True)
        self.log("val/comp_loss", results["comp_loss"], on_step=False, on_epoch=True)
        # self.log("val/sem_loss", results["sem_loss"], on_step=False, on_epoch=True)

        self.log("val/exp_ppl", results["exp_ppl"], on_step=False, on_epoch=True)
        # fmt: on

        # exp
        exp_logits = argmax(results["exp_logits"], dim=-1)
        exp_predictions = self.arch.ids_to_clean_text(exp_logits)
        exp_targets = self.arch.ids_to_clean_text(batch["story_ids"])

        # comp
        comp_logits = argmax(results["comp_logits"], dim=-1)
        comp_predictions = self.arch.ids_to_clean_text(comp_logits)
        comp_targets = self.arch.ids_to_clean_text(batch["summary_ids"])

        metrics = {
            # exp
            **bleu(exp_targets, exp_predictions, prefix="exp_"),
            **distinct_n(exp_predictions, prefix="exp_"),
            # comp
            "comp_bleu": bleu(comp_targets, comp_predictions, prefix="comp_")["comp_bleu"],
        }

        for k, v in metrics.items():
            metrics[k] = tensor(v, device=self.device)

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
        self.log("test/exp_loss", results["exp_loss"], on_step=False, on_epoch=True)
        self.log("test/comp_loss", results["comp_loss"], on_step=False, on_epoch=True)
        # self.log("test/sem_loss", results["sem_loss"], on_step=False, on_epoch=True)

        self.log("test/exp_ppl", results["exp_ppl"], on_step=False, on_epoch=True)
        # fmt: on

        # exp: accumulate bleu sys & refs
        story_targets = self.arch.expander.ids_to_clean_text(batch["story_ids"])
        story_predictions = self.arch.expander.generate_from_ids(
            {"input_ids": batch["summary_ids"], "attention_mask": batch["summary_attn_msk"]}
        )

        self.story_references.extend(story_targets)
        self.story_predictions.extend(story_predictions)

        self.bleurt.add_batch(predictions=story_predictions, references=story_targets)
        self.bert_score.add_batch(predictions=story_predictions, references=story_targets)

        # comp: accumulate bleu sys & refs
        summary_targets = self.arch.compressor.ids_to_clean_text(batch["summary_ids"])
        summary_predictions = self.arch.compressor.generate_from_ids(
            {"input_ids": batch["story_ids"], "attention_mask": batch["story_attn_msk"]}
        )

        self.summary_references.extend(summary_targets)
        self.summary_predictions.extend(summary_predictions)

        return results["loss"]

    def on_test_epoch_end(self):
        """
        Calculate corpus-level metrics on targets and
            model-generated predictions
        """

        metrics = {
            # exp
            **bleu(self.story_references, self.story_predictions, prefix="exp_"),
            **distinct_n(self.story_predictions, prefix="exp_"),
            "bleurt": np.mean(self.bleurt.compute()["scores"]),
            "bertscore": np.mean(self.bertscore.compute(lang="en", rescale_with_baseline=True)["f1"]),
            # comp
            "comp_bleu": bleu(self.summary_references, self.summary_predictions, prefix="comp_")["comp_bleu"],
        }

        self.log_at_val_test_epoch_end(metrics, "test")

        self.story_predictions, self.story_references = [], []
        self.summary_predictions, self.summary_references = [], []

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.arch.generate(conditioning_sentences)

    def configure_optimizers(self):
        # noinspection PyTypeChecker
        return AdamW(
            [
                {"params": self.arch.compressor.parameters(), "lr": self.comp_lr},
                {"params": self.arch.expander.parameters(), "lr": self.exp_lr},
            ]
        )

    @property
    def tokenizer(self) -> BartTokenizerFast:
        return self.arch.tokenizer

from typing import Dict, List, Optional, Union

from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import Expander


class ExpanderModel(LightningModule):
    def __init__(self, model_name_or_path: str, learning_rate: float = 5e-5, max_generation_length: int = 70, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.arch = Expander(
            model_name_or_path=model_name_or_path,
            max_generation_length=max_generation_length,
        )
        self.lr = learning_rate

    def forward(self, dict_input: Dict) -> Dict:
        return self.arch(dict_input)

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log("train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", results["accuracy"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bleu", results["bleu"], on_step=True, on_epoch=True, prog_bar=True)
        # fmt: on

        return results["loss"]

    def val_test_step(self, batch, prefix: str) -> Optional[Dict]:
        results = self.forward(batch)

        # fmt: off
        self.log(f"{prefix}/loss", results["loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/acc", results["accuracy"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/bleu", results["bleu"], on_step=False, on_epoch=True)
        # fmt: on

        return results["loss"]

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, "test")

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> List[str]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.arch.generate(conditioning_sentences)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self.arch.tokenizer

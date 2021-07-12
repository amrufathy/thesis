from typing import Dict, List, Union

from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import Expander


class ExpanderModel(LightningModule):
    def __init__(
        self,
        name: str,
        model_name_or_path: str,
        learning_rate: float = 5e-5,
        max_generation_length: int = 70,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Expander(
            model_name_or_path=model_name_or_path,
            max_generation_length=max_generation_length,
        )
        self.lr = learning_rate

    def forward(self, dict_input: Dict) -> Dict:
        return self.model(dict_input)

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log(f"train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc", results["accuracy"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/bleu", results["bleu"], on_step=True, on_epoch=True, prog_bar=True)
        # fmt: on

        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log(f"val/loss", results["loss"], on_step=False, on_epoch=True)
        self.log(f"val/acc", results["accuracy"], on_step=False, on_epoch=True)
        self.log(f"val/bleu", results["bleu"], on_step=False, on_epoch=True)
        # fmt: on

        return results["loss"]

    def test_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log(f"val/loss", results["loss"], on_step=False, on_epoch=True)
        self.log(f"val/acc", results["accuracy"], on_step=False, on_epoch=True)
        self.log(f"val/bleu", results["bleu"], on_step=False, on_epoch=True)
        # fmt: on

        return results["loss"]

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> List[str]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.model.generate(conditioning_sentences)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self.model.tokenizer

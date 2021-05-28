from typing import Dict, List, Optional, Union

from pytorch_lightning import LightningModule
from transformers.optimization import AdamW

from src.models.modules import Expander


class ExpanderModel(LightningModule):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.save_hyperparameters()

        self.model = Expander(model_name_or_path=model_name_or_path)

    def forward(self, dict_input: Dict) -> Dict:
        return self.model(dict_input)

    def step(self, batch, prefix: str) -> Optional[Dict]:
        results = self.forward(batch)
        loss = results["loss"]

        # fmt: off
        self.log(f"{prefix}/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", results["accuracy"], on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"{prefix}/bleu", results["bleu"], on_step=True, on_epoch=True, prog_bar=False)
        # fmt: on

        del results

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> List[str]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.model.generate(conditioning_sentences)

    def configure_optimizers(self):
        return AdamW(self.model.parameters())

from typing import Dict, Optional

from pytorch_lightning import LightningModule
from src.models.modules import CycleArchitecture
from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW

# TODO: print out generated stories [Callback]
# TODO: check if cycle works


class CycleModel(LightningModule):
    def __init__(
        self,
        name: str,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool,
        learning_rate: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CycleArchitecture(
            expander_model_name=expander_model_name,
            compressor_model_name=compressor_model_name,
            use_gumbel_softmax=use_gumbel_softmax,
        )
        self.lr = learning_rate

    def forward(self, dict_input: Dict) -> Dict:
        return self.model(dict_input)

    def step(self, batch, prefix: str) -> Optional[Dict]:
        results = self.forward(batch)
        loss = results["loss"]

        # fmt: off
        self.log(f"{prefix}/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/exp_loss", results["exp_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/comp_loss", results["comp_loss"], on_step=True, on_epoch=True, prog_bar=True)

        self.log(f"{prefix}/acc", results["acc"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/exp_acc", results["exp_acc"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/comp_acc", results["comp_acc"], on_step=True, on_epoch=True)

        self.log(f"{prefix}/bleu", results["bleu"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/exp_bleu", results["exp_bleu"], on_step=True, on_epoch=True)
        self.log(f"{prefix}/comp_bleu", results["comp_bleu"], on_step=True, on_epoch=True)
        # fmt: on

        del results

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self.model.tokenizer

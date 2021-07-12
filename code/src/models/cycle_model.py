from typing import Dict, List, Optional, Tuple, Union

from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW

from src.models.modules import CycleArchitecture

# TODO: use AutoModel & AutoTokenizer APIs


class CycleModel(LightningModule):
    def __init__(
        self,
        name: str,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool = False,
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

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)

        # fmt: off
        self.log(f"train/loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/rec_loss", results["rec_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/exp_loss", results["exp_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/comp_loss", results["comp_loss"], on_step=True, on_epoch=True, prog_bar=True)

        self.log(f"train/acc", results["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/exp_acc", results["exp_acc"], on_step=True, on_epoch=True)
        self.log(f"train/comp_acc", results["comp_acc"], on_step=True, on_epoch=True)

        self.log(f"train/bleu", results["bleu"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/exp_bleu", results["exp_bleu"], on_step=True, on_epoch=True)
        self.log(f"train/comp_bleu", results["comp_bleu"], on_step=True, on_epoch=True)
        # fmt: on

        return results["loss"]

    def val_test_step(self, batch, prefix: str) -> Optional[Dict]:
        results = self.forward(batch)

        # fmt: off
        self.log(f"{prefix}/loss", results["loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/rec_loss", results["rec_loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/exp_loss", results["exp_loss"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/comp_loss", results["comp_loss"], on_step=False, on_epoch=True)

        self.log(f"{prefix}/acc", results["acc"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/exp_acc", results["exp_acc"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/comp_acc", results["comp_acc"], on_step=False, on_epoch=True)

        self.log(f"{prefix}/bleu", results["bleu"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/exp_bleu", results["exp_bleu"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/comp_bleu", results["comp_bleu"], on_step=False, on_epoch=True)
        # fmt: on

        return results["loss"]

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, "test")

    def generate(self, conditioning_sentences: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
        if isinstance(conditioning_sentences, str):
            conditioning_sentences = [conditioning_sentences]

        return self.model.generate(conditioning_sentences)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self.model.tokenizer

from pytorch_lightning import LightningModule
from typing import Dict, Optional
from src.models.modules import CycleArchitecture
from transformers.optimization import AdamW


# TODO: print out generated stories (done)
# TODO: metrics: add accuracy/sacre-bleu/rest of losses (done)
# TODO: split data to train/val/test (done)
# TODO: add train_step/val_step/test_step logging (done)
# TODO: check that each one of the models works first (done)
# TODO: check if cycle works


class CycleModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # self.save_hyperparameters()

        self.model = CycleArchitecture()

        # self.metric_hist = {
        #     'train/loss': [],
        #     'val/loss': []
        # }

    def forward(self, dict_input: Dict) -> Dict:
        return self.model(dict_input)

    def step(self, batch, prefix: str) -> Optional[Dict]:
        results = self.forward(batch)
        loss = results['loss']

        self.log(f'{prefix}/loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/exp_loss', results['exp_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/comp_loss', results['comp_loss'], on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{prefix}/acc', results['acc'], on_step=True, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/exp_acc', results['exp_acc'], on_step=True, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/comp_acc', results['comp_acc'], on_step=True, on_epoch=True, prog_bar=False)

        self.log(f'{prefix}/bleu', results['bleu'], on_step=True, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/exp_bleu', results['exp_bleu'], on_step=True, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/comp_bleu', results['comp_bleu'], on_step=True, on_epoch=True, prog_bar=False)

        del results

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    # def training_epoch_end(self, outputs: List[Any]):
    #     self.metric_hist['train/loss'].append(self.trainer.callback_metrics['train/loss'])
    #     self.log('train/best_loss', min(self.metric_hist['train/loss']))

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    # def validation_epoch_end(self, outputs: List[Any]):
    #     self.metric_hist['val/loss'].append(self.trainer.callback_metrics['val/loss'])
    #     self.log('val/best_loss', min(self.metric_hist['val/loss']))

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        return AdamW(self.model.parameters())

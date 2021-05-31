from typing import Dict

from sacrebleu import corpus_bleu
from torch import argmax, cuda, device, nn, tensor
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander


class CycleArchitecture(nn.Module):
    def __init__(self, expander_model_name: str, compressor_model_name: str):
        super().__init__()

        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.expander = Expander(
            model_name_or_path=expander_model_name, tokenizer=self.tokenizer
        )
        self.compressor = Compressor(
            model_name_or_path=compressor_model_name, tokenizer=self.tokenizer
        )
        self.device = device("cuda") if cuda.is_available() else device("cpu")

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through the whole cycle
        input -> expander -> intermediate output -> compressor -> reconstructed input

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # INFO - Step 1: Expansion (Summary -> Story)
        expansion_results = self.expander(dict_input)
        expansion_loss, expansion_logits = (
            expansion_results["loss"],
            expansion_results["logits"],
        )
        expansion_accuracy, expansion_bleu = (
            expansion_results["accuracy"],
            expansion_results["bleu"],
        )

        # overwrite dict_input['story_ids'] (original story ids) with generated_story_ids
        generated_story_ids = argmax(
            expansion_logits, dim=-1
        )  # TODO: use gumbel softmax
        dict_input["story_ids"] = generated_story_ids

        del expansion_results, generated_story_ids

        # INFO - Step 2: Compression (Generated Story -> Reconstructed Summary)
        compression_results = self.compressor(dict_input)
        compression_loss, compression_logits = (
            compression_results["loss"],
            compression_results["logits"],
        )
        compression_accuracy, compression_bleu = (
            compression_results["accuracy"],
            compression_results["bleu"],
        )

        # INFO - Step 3: Calculate Reconstruction Loss
        # ignore <s> and reshape
        reconstructed_logits = (
            compression_logits[:, 1:].contiguous().view(-1, compression_logits.size(-1))
        )
        summary_labels = dict_input["summary_labels"][:, 1:].contiguous().view(-1)

        reconstruction_loss = cross_entropy(reconstructed_logits, summary_labels)
        total_loss = expansion_loss + compression_loss + reconstruction_loss

        del compression_results, reconstructed_logits, summary_labels

        # reconstruction accuracy
        reconstructed_summary_ids = argmax(compression_logits, dim=-1)
        masked_labels = dict_input["summary_labels"].detach().clone()
        masked_labels[
            masked_labels[:, :] == -100
        ] = self.tokenizer.pad_token_id  # restore padding token id

        reconstruction_accuracy = accuracy(reconstructed_summary_ids, masked_labels)
        total_accuracy = (
            expansion_accuracy + compression_accuracy + reconstruction_accuracy
        )

        # reconstruction bleu
        # TODO: (run in val only)
        predictions = self.tokenizer.batch_decode(
            reconstructed_summary_ids, skip_special_tokens=True
        )
        references = self.tokenizer.batch_decode(
            masked_labels, skip_special_tokens=True
        )

        bleu = corpus_bleu(predictions, [references])

        del reconstructed_summary_ids, masked_labels, predictions, references

        return {
            # losses
            "loss": total_loss,
            "exp_loss": expansion_loss,
            "comp_loss": compression_loss,
            # accuracy
            "acc": total_accuracy,
            "exp_acc": expansion_accuracy,
            "comp_acc": compression_accuracy,
            # bleu
            "bleu": tensor(bleu.score, device=self.device),
            "exp_bleu": expansion_bleu,
            "comp_bleu": compression_bleu,
        }

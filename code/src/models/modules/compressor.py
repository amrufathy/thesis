from typing import Dict, List

from sacrebleu import corpus_bleu
from torch import argmax, cuda, device, nn, tensor
from torchmetrics.functional import accuracy
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right


class Compressor(nn.Module):
    def __init__(self, model_name_or_path: str, tokenizer: BartTokenizerFast = None):
        super().__init__()

        self.compressor: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.device = device("cuda") if cuda.is_available() else device("cpu")

        if not tokenizer:
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = tokenizer

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through compressor bart
        input story -> compressor -> generated summary

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # shift `decoder ids` & `mask` to the right
        summary_ids_shifted = shift_tokens_right(
            dict_input["summary_ids"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        # summary_msk_shifted = shift_tokens_right(dict_input['summary_attn_msk'], 0, 1)

        # feed the model
        compression_results = self.compressor(
            **{
                "input_ids": dict_input["story_ids"],
                "attention_mask": dict_input["story_attn_msk"],  # TODO: check if right in case of used within cycle
                "decoder_input_ids": summary_ids_shifted,
                "decoder_attention_mask": dict_input["summary_attn_msk"],
                "labels": dict_input["summary_labels"],
            },
            use_cache=False
        )

        compression_loss, compression_logits = (
            compression_results.loss,
            compression_results.logits,
        )

        del summary_ids_shifted, compression_results

        # compute metrics

        # accuracy
        generated_summary_ids = argmax(compression_logits, dim=-1)
        masked_labels = dict_input["summary_labels"].detach().clone()
        masked_labels[masked_labels[:, :] == -100] = self.tokenizer.pad_token_id  # restore padding token id
        acc = accuracy(generated_summary_ids, masked_labels)

        # bleu
        predictions = self.tokenizer.batch_decode(generated_summary_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(masked_labels, skip_special_tokens=True)
        # predictions = self.adjust_padding(predictions, references)

        bleu = corpus_bleu(predictions, [references])

        del generated_summary_ids, masked_labels, predictions, references

        return {
            "loss": compression_loss,
            "logits": compression_logits,
            "accuracy": acc,
            "bleu": tensor(bleu.score, device=self.device),
        }

    def generate(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate summaries depending on conditional input stories
        """

        tokenized_sentences = self.tokenizer(conditioning_sentences, padding="longest", return_tensors="pt")
        generated_ids = self.compressor.generate(**tokenized_sentences)
        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_summaries

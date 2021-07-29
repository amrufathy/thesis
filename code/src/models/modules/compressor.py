from typing import Dict, List

from sacrebleu import corpus_bleu
from torch import argmax, cuda, device, nn, tensor
from torchmetrics.functional import accuracy
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right


class Compressor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: BartTokenizerFast = None,
        max_generation_length: int = 7,
    ):
        super().__init__()

        self.compressor: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.max_generation_length = max_generation_length

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

        # shift `decoder ids` to the right
        summary_ids_shifted = shift_tokens_right(
            dict_input["summary_ids"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        # feed the model
        feed_dict = {
            "attention_mask": dict_input["story_attn_msk"],  # TODO: check if right in case of used within cycle
            "decoder_input_ids": summary_ids_shifted,
            "decoder_attention_mask": dict_input["summary_attn_msk"],
            "labels": dict_input["summary_labels"],
        }

        if "story_embs" in dict_input:
            feed_dict["inputs_embeds"] = dict_input["story_embs"]
        else:
            feed_dict["input_ids"] = dict_input["story_ids"]

        compression_results = self.compressor(**feed_dict, use_cache=False)

        compression_loss, compression_logits = (
            compression_results.loss,
            compression_results.logits,
        )

        del summary_ids_shifted, compression_results, feed_dict

        # compute metrics

        # accuracy
        generated_summary_ids = argmax(compression_logits, dim=-1)
        masked_labels = dict_input["summary_labels"].detach().clone()
        masked_labels[masked_labels[:, :] == -100] = self.tokenizer.pad_token_id  # restore padding token id
        acc = accuracy(generated_summary_ids, masked_labels)

        # bleu
        predictions = self.tokenizer.batch_decode(generated_summary_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(masked_labels, skip_special_tokens=True)

        bleu = corpus_bleu(predictions, [references])

        del generated_summary_ids, masked_labels, predictions, references

        return {
            "loss": compression_loss,
            "logits": compression_logits.detach(),
            "accuracy": acc.detach(),
            "bleu": tensor(bleu.score, device=self.device).detach(),
        }

    def generate(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate summaries depending on conditional input stories
        """

        tokenized_sentences = self.tokenizer(conditioning_sentences, padding="longest", return_tensors="pt")
        generated_ids = self.compressor.generate(
            **tokenized_sentences,
            num_beams=5,
            do_sample=True,
            early_stopping=False,
            min_length=2,
            max_length=self.max_generation_length
        )
        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_summaries

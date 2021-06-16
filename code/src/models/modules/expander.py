from typing import Dict, List

from sacrebleu import corpus_bleu
from torch import argmax, cuda, device, nn, tensor
from torchmetrics.functional import accuracy
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right


class Expander(nn.Module):
    def __init__(self, model_name_or_path: str, tokenizer: BartTokenizerFast = None):
        super().__init__()

        self.expander: BartForConditionalGeneration = (
            BartForConditionalGeneration.from_pretrained(model_name_or_path)
        )
        self.device = device("cuda") if cuda.is_available() else device("cpu")

        if not tokenizer:
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = tokenizer

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through expander bart
        input summary -> expander -> generated story

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # shift `decoder ids` & `mask` to the right
        story_ids_shifted = shift_tokens_right(
            dict_input["story_ids"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )
        # story_msk_shifted = shift_tokens_right(dict_input['story_attn_msk'], 0, 1)

        # feed the model
        expansion_results = self.expander(
            **{
                "input_ids": dict_input["summary_ids"],
                "attention_mask": dict_input["summary_attn_msk"],
                "decoder_input_ids": story_ids_shifted,
                "decoder_attention_mask": dict_input["story_attn_msk"],
                "labels": dict_input["story_labels"],
            },
            use_cache=False
        )

        expansion_loss, expansion_logits = (
            expansion_results.loss,
            expansion_results.logits,
        )

        del story_ids_shifted, expansion_results

        # compute metrics

        # accuracy
        generated_story_ids = argmax(expansion_logits, dim=-1)
        masked_labels = dict_input["story_labels"].detach().clone()
        masked_labels[
            masked_labels[:, :] == -100
        ] = self.tokenizer.pad_token_id  # restore padding token id
        acc = accuracy(generated_story_ids, masked_labels)

        # bleu
        predictions = self.tokenizer.batch_decode(
            generated_story_ids, skip_special_tokens=True
        )
        references = self.tokenizer.batch_decode(
            masked_labels, skip_special_tokens=True
        )
        # predictions = self.adjust_padding(predictions, references)

        bleu = corpus_bleu(predictions, [references])

        del generated_story_ids, masked_labels, predictions, references

        return {
            "loss": expansion_loss,
            "logits": expansion_logits,
            "accuracy": acc,
            "bleu": tensor(bleu.score, device=self.device),
        }

    def generate(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate stories depending on conditional input summaries
        """

        tokenized_sentences = self.tokenizer(
            conditioning_sentences, padding="longest", return_tensors="pt"
        )
        generated_ids = self.expander.generate(**tokenized_sentences)
        generated_stories = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_stories

    def adjust_padding(self, predictions, references):
        """
        Adjusts the token-wise padding of `predictions` to match
            lengths of `references`
        """

        adjusted_paddings = []
        for p, r in zip(predictions, references):
            tp = self.tokenizer.tokenize(self.tokenizer.clean_up_tokenization(p))
            tr = self.tokenizer.tokenize(self.tokenizer.clean_up_tokenization(r))

            if len(tp) == len(tr):
                adjusted_paddings.append(p)
            elif len(tp) > len(tr):
                ts = tp[: len(tr)]
                adjusted_paddings.append(self.tokenizer.convert_tokens_to_string(ts))
            else:  # len(tp) < len(tr)
                # Ġ character represents a whitespace in the byte-level representation
                ts = tp + ["Ġ" + self.tokenizer.pad_token] * (len(tr) - len(tp))
                adjusted_paddings.append(self.tokenizer.convert_tokens_to_string(ts))

            del tp, tr

        return adjusted_paddings

from typing import Dict, List

from sacrebleu import corpus_bleu
from torch import argmax, cuda, device, exp, nn, tensor
from torchmetrics.functional import accuracy
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.utils.model_utils import distinct_n


class Expander(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: BartTokenizerFast = None,
        max_generation_length: int = 70,
    ):
        super().__init__()

        self.expander: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.max_generation_length = max_generation_length

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

        # shift `decoder ids` to the right
        story_ids_shifted = shift_tokens_right(
            dict_input["story_ids"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        # feed the model
        feed_dict = {
            "attention_mask": dict_input["summary_attn_msk"],
            "decoder_input_ids": story_ids_shifted,
            "decoder_attention_mask": dict_input["story_attn_msk"],
            "labels": dict_input["story_labels"],
        }

        if "summary_embs" in dict_input:
            feed_dict["inputs_embeds"] = dict_input["summary_embs"]
        else:
            feed_dict["input_ids"] = dict_input["summary_ids"]

        expansion_results = self.expander(**feed_dict, use_cache=False)

        expansion_loss, expansion_logits = (
            expansion_results.loss,
            expansion_results.logits,
        )

        del story_ids_shifted, expansion_results, feed_dict

        # compute metrics

        # accuracy
        generated_story_ids = argmax(expansion_logits, dim=-1)
        masked_labels = dict_input["story_labels"].detach().clone()
        masked_labels[masked_labels[:, :] == -100] = self.tokenizer.pad_token_id  # restore padding token id
        acc = accuracy(generated_story_ids, masked_labels)

        # bleu
        predictions = self.tokenizer.batch_decode(generated_story_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(masked_labels, skip_special_tokens=True)

        bleu = corpus_bleu(predictions, [references])
        bleu_aggr = tensor(bleu.score, device=self.device).detach()
        bleu1 = tensor(bleu.precisions[0], device=self.device).detach()
        bleu2 = tensor(bleu.precisions[1], device=self.device).detach()
        bleu3 = tensor(bleu.precisions[2], device=self.device).detach()
        bleu4 = tensor(bleu.precisions[3], device=self.device).detach()

        # distinctness
        distinct1 = tensor(distinct_n(predictions, ngram_size=1), device=self.device).detach()
        distinct2 = tensor(distinct_n(predictions, ngram_size=2), device=self.device).detach()
        distinct3 = tensor(distinct_n(predictions, ngram_size=3), device=self.device).detach()
        distinct4 = tensor(distinct_n(predictions, ngram_size=4), device=self.device).detach()

        # perplexity
        perplexity = exp(expansion_loss).detach()

        del generated_story_ids, masked_labels, predictions, references

        return {
            "loss": expansion_loss,
            "logits": expansion_logits.detach(),
            "accuracy": acc.detach(),
            "bleu": bleu_aggr,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "ppl": perplexity,
            "distinct1": distinct1,
            "distinct2": distinct2,
            "distinct3": distinct3,
            "distinct4": distinct4,
        }

    def generate(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate stories depending on conditional input summaries
        """

        tokenized_sentences = self.tokenizer(conditioning_sentences, padding="longest", return_tensors="pt")
        generated_ids = self.expander.generate(
            **tokenized_sentences,
            num_beams=5,
            do_sample=True,
            early_stopping=False,
            top_p=0.9,
            min_length=self.max_generation_length // 2,
            max_length=self.max_generation_length,
            length_penalty=1.5
        )
        generated_stories = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_stories

    def get_embeddings(self):
        return self.expander.get_input_embeddings().weight

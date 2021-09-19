from typing import Dict, List, Union

from torch import exp, nn
from transformers import BartForConditionalGeneration, BartTokenizerFast, BatchEncoding
from transformers.models.bart.modeling_bart import shift_tokens_right


class Expander(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: BartTokenizerFast = None,
        max_generation_length: int = 70,
    ):
        super().__init__()

        self.tokenizer: BartTokenizerFast = (
            tokenizer if tokenizer is not None else BartTokenizerFast.from_pretrained(model_name_or_path)
        )

        self.expander: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.expander.resize_token_embeddings(len(self.tokenizer))
        self.max_generation_length = max_generation_length

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through expander bart
        input summary -> expander -> generated story

        @param dict_input: contains input_ids, attention_masks for both story and summary
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
        }

        # if using Gumbel-Softmax, feed embedding directly
        if "summary_embs" in dict_input:
            feed_dict["inputs_embeds"] = dict_input["summary_embs"]
        else:
            feed_dict["input_ids"] = dict_input["summary_ids"]

        expansion_results = self.expander(**feed_dict, use_cache=False)

        loss = self.loss_fn(expansion_results.logits.permute(0, 2, 1), dict_input["story_ids"])

        return {"loss": loss, "ppl": exp(loss).detach().item(), "logits": expansion_results.logits.detach()}

    # INFO - generate functions used for inference

    def generate_from_text(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate stories depending on conditional text input summaries
        """

        conditioning_sentences = self.tokenizer(conditioning_sentences, padding="longest", return_tensors="pt")
        return self.generate_from_ids(conditioning_sentences)

    def generate_from_ids(self, conditioning_sentences: Union[Dict, BatchEncoding]) -> List[str]:
        """
        Generate stories depending on conditional tokenized input summaries
        """

        generated_ids = self.expander.generate(
            **conditioning_sentences,
            num_beams=5,
            do_sample=True,
            early_stopping=False,
            top_k=50,
            top_p=0.9,
            temperature=1.0,  # default 1.0
            max_length=self.max_generation_length,
        )

        return self.ids_to_clean_text(generated_ids)

    def ids_to_clean_text(self, generated_ids) -> List[str]:
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return list(map(str.strip, generated_text))

    @property
    def input_embeddings(self):
        return self.expander.get_input_embeddings().weight

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

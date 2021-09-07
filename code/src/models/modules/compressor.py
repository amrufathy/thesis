from typing import Dict, List, Union

from torch import nn
from transformers import BartForConditionalGeneration, BartTokenizerFast, BatchEncoding
from transformers.models.bart.modeling_bart import shift_tokens_right


class Compressor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: BartTokenizerFast = None,
        max_generation_length: int = 7,
    ):
        super().__init__()

        self.tokenizer: BartTokenizerFast = (
            tokenizer if tokenizer is not None else BartTokenizerFast.from_pretrained(model_name_or_path)
        )

        self.compressor: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.compressor.resize_token_embeddings(len(self.tokenizer))
        self.max_generation_length = max_generation_length

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through compressor bart
        input story -> compressor -> generated summary

        @param dict_input: contains input_ids, attention_masks for both story and summary
        """

        # shift `decoder ids` to the right
        summary_ids_shifted = shift_tokens_right(
            dict_input["summary_ids"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        # feed the model
        feed_dict = {
            "attention_mask": dict_input["story_attn_msk"],
            "decoder_input_ids": summary_ids_shifted,
            "decoder_attention_mask": dict_input["summary_attn_msk"],
        }

        # if using Gumbel-Softmax, feed embedding directly
        if "story_embs" in dict_input:
            feed_dict["inputs_embeds"] = dict_input["story_embs"]
        else:
            feed_dict["input_ids"] = dict_input["story_ids"]

        compression_results = self.compressor(**feed_dict, use_cache=False)

        loss = self.loss_fn(compression_results.logits.permute(0, 2, 1), dict_input["summary_ids"])

        return {"loss": loss, "logits": compression_results.logits.detach()}

    # INFO - generate functions used for inference

    def generate_from_text(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate summaries depending on conditional text input stories
        """

        conditioning_sentences = self.tokenizer(conditioning_sentences, padding="longest", return_tensors="pt")
        return self.generate_from_ids(conditioning_sentences)

    def generate_from_ids(self, conditioning_sentences: Union[Dict, BatchEncoding]) -> List[str]:
        """
        Generate summaries depending on conditional text input stories
        """

        generated_ids = self.compressor.generate(
            **conditioning_sentences,
            num_beams=5,
            do_sample=True,
            early_stopping=False,
            top_k=50,
            top_p=0.9,
            max_length=self.max_generation_length
        )

        return self.ids_to_clean_text(generated_ids)

    def ids_to_clean_text(self, generated_ids) -> List[str]:
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return list(map(str.strip, generated_text))

    @property
    def input_embeddings(self):
        return self.compressor.get_input_embeddings().weight

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

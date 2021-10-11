from typing import Dict, List, Tuple

from torch import argmax, nn
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander, SemanticBert
from src.utils.model_utils import get_gumbel_sampled_embeddings


class CycleArchitectureCompress(nn.Module):
    def __init__(
        self,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool = False,
        max_story_length: int = 70,
        max_summary_length: int = 7,
        **kwargs
    ):
        super().__init__()

        assert expander_model_name == compressor_model_name

        self.tokenizer = BartTokenizerFast.from_pretrained(expander_model_name)
        self.expander = Expander(
            model_name_or_path=expander_model_name, tokenizer=self.tokenizer, max_generation_length=max_story_length
        )
        self.compressor = Compressor(
            model_name_or_path=compressor_model_name, tokenizer=self.tokenizer, max_generation_length=max_summary_length
        )
        self.use_gumbel_softmax = use_gumbel_softmax

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.expander.pad_token_id)

        # self.semantic_bert = SemanticBert(**kwargs)
        # self.use_semantic_sim = kwargs.get("use_semantic_similarity", False)

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through the whole cycle
        input -> expander -> intermediate output -> compressor -> reconstructed input

        @param dict_input: contains input_ids, attention_masks for both story and summary
        """

        dict_input = dict_input.copy()

        # INFO - Step 1: Expansion (Summary -> Generated Story)
        expansion_results = self.expander(dict_input)
        expansion_loss, expansion_logits, expansion_ppl = (
            expansion_results["loss"],
            expansion_results["logits"],
            expansion_results["ppl"],
        )

        # INFO: learn semantic similarity
        # if self.use_semantic_sim:
        #     semantic_loss = self.semantic_bert.step(
        #         gold_stories_ids=dict_input["story_ids"],
        #         generated_stories_ids=argmax(expansion_logits, dim=-1),
        #         external_tokenizer=self.tokenizer,
        #     )
        # else:
        #     semantic_loss = 0.0

        # INFO: if using gumbel then the whole cycle is differentiable
        #  if not using gumbel then dual learning technique
        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(expansion_logits, self.compressor.input_embeddings)

            # pass generated story embeddings to compressor
            dict_input["story_embs"] = embs
        else:
            generated_story_ids = argmax(expansion_logits, dim=-1)

            # overwrite original story ids with `generated_story_ids`
            dict_input["story_ids"] = generated_story_ids

        # INFO - Step 2: Compression (Generated Story -> Reconstructed Summary)
        compression_results = self.compressor(dict_input)
        compression_loss, compression_logits = (
            compression_results["loss"],
            compression_results["logits"],
        )

        # INFO - Step 3: Calculate Aggregated Metrics
        total_loss = expansion_loss + compression_loss

        return {
            # losses
            "loss": total_loss,
            "exp_loss": expansion_loss,
            "comp_loss": compression_loss,
            # "sem_loss": semantic_loss,
            # logits
            "exp_logits": expansion_logits,
            "comp_logits": compression_logits,
            # perplexity
            "exp_ppl": expansion_ppl,
        }

    def generate_from_text(self, conditioning_sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate intermediate stories and reconstructed summaries based on
            conditional text input summaries
        """

        generated_stories = self.expander.generate_from_text(conditioning_sentences)
        reconstructed_summaries = self.compressor.generate_from_text(generated_stories)

        return generated_stories, reconstructed_summaries

    def ids_to_clean_text(self, generated_ids) -> List[str]:
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return list(map(str.strip, generated_text))

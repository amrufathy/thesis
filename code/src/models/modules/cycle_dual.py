from typing import Dict, List, Tuple

from torch import argmax, mean, nn, stack, tensor
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander, SemanticBert
from src.utils.model_utils import get_gumbel_sampled_embeddings


class CycleArchitectureDual(nn.Module):
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
        runs the input through the whole cycle in both directions

        1) input -> expander -> intermediate output -> compressor -> reconstructed input
        2) input -> compressor -> intermediate output -> expander -> reconstructed input

        @param dict_input: contains input_ids, attention_masks for both story and summary
        """

        # ==============================================
        # ==============================================
        # INFO: First Direction (Expander - Compressor)
        # ==============================================
        # ==============================================

        dict_input = dict_input.copy()
        original_input = dict_input.copy()

        # INFO - Step 1: Expansion (Summary -> Generated Story)
        expansion_results_1 = self.expander(dict_input)
        expansion_loss_1, expansion_logits_1, expansion_ppl_1 = (
            expansion_results_1["loss"],
            expansion_results_1["logits"],
            expansion_results_1["ppl"],
        )

        # INFO: learn semantic similarity
        # if self.use_semantic_sim:
        #     semantic_loss_1 = self.semantic_bert.step(
        #         gold_stories_ids=original_input["story_ids"],
        #         generated_stories_ids=argmax(expansion_logits_1, dim=-1),
        #         external_tokenizer=self.tokenizer,
        #     )
        # else:
        #     semantic_loss_1 = 0.0

        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(expansion_logits_1, self.compressor.input_embeddings)

            # pass generated story embeddings to compressor
            dict_input["story_embs"] = embs
        else:
            generated_story_ids = argmax(expansion_logits_1, dim=-1)

            # overwrite original story ids with `generated_story_ids`
            dict_input["story_ids"] = generated_story_ids

        # INFO - Step 2: Compression (Generated Story -> Reconstructed Summary)
        compression_results_1 = self.compressor(dict_input)
        compression_loss_1, compression_logits_1 = (
            compression_results_1["loss"],
            compression_results_1["logits"],
        )

        del expansion_results_1, compression_results_1

        # restore dict_input
        dict_input["story_ids"] = original_input["story_ids"]
        if self.use_gumbel_softmax:
            dict_input.pop("story_embs")

        # ==============================================
        # ==============================================
        # INFO: Second Direction (Compressor - Expander)
        # ==============================================
        # ==============================================

        # INFO - Step 1: Compression (Story -> Generated Summary)
        compression_results_2 = self.compressor(dict_input)
        compression_loss_2, compression_logits_2 = (
            compression_results_2["loss"],
            compression_results_2["logits"],
        )

        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(compression_logits_2, self.expander.get_embeddings())

            # pass generated summary embeddings to compressor
            dict_input["summary_embs"] = embs
        else:
            generated_summary_ids = argmax(compression_logits_2, dim=-1)

            # overwrite original summary ids with `generated_summary_ids`
            dict_input["summary_ids"] = generated_summary_ids

        # INFO - Step 2: Expansion (Generated Summary -> Reconstructed Story)
        expansion_results_2 = self.expander(dict_input)
        expansion_loss_2, expansion_logits_2, expansion_ppl_2 = (
            expansion_results_2["loss"],
            expansion_results_2["logits"],
            expansion_results_2["ppl"],
        )

        # INFO: learn semantic similarity
        # if self.use_semantic_sim:
        #     semantic_loss_2 = self.semantic_bert.step(
        #         gold_stories_ids=original_input["story_ids"],
        #         generated_stories_ids=argmax(expansion_logits_2, dim=-1),
        #         external_tokenizer=self.tokenizer,
        #     )
        # else:
        #     semantic_loss_2 = 0.0

        del compression_results_2, expansion_results_2

        # ==============================================
        # ==============================================
        # INFO - Step 3: Calculate Aggregated Metrics
        # ==============================================
        # ==============================================

        # loss has to be sum (not mean) to be differentiable
        expansion_loss = expansion_loss_1 + expansion_loss_2
        expansion_ppl = mean(tensor([expansion_ppl_1, expansion_ppl_2]))

        compression_loss = compression_loss_1 + compression_loss_2

        # semantic_loss = semantic_loss_1 + semantic_loss_2

        total_loss = expansion_loss + compression_loss

        expansion_logits = mean(stack([expansion_logits_1, expansion_logits_2], dim=-1), dim=-1)
        compression_logits = mean(stack([compression_logits_1, compression_logits_2], dim=-1), dim=-1)

        return {
            # losses
            "loss": total_loss,
            "exp_loss": expansion_loss,
            "comp_loss": compression_loss,
            # "sem_loss": semantic_loss,
            # logits
            "exp_logits": expansion_logits,
            "comp_logits": compression_logits,
            # extra exp metric
            "exp_ppl": expansion_ppl.detach().item(),
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

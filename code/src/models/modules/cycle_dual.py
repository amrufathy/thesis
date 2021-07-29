from typing import Dict, List, Tuple

from torch import argmax, cuda, device, mean, nn, tensor
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander
from src.utils.model_utils import get_gumbel_sampled_embeddings


class CycleArchitectureDual(nn.Module):
    def __init__(
        self,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool = False,
    ):
        super().__init__()

        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.expander = Expander(model_name_or_path=expander_model_name, tokenizer=self.tokenizer)
        self.compressor = Compressor(model_name_or_path=compressor_model_name, tokenizer=self.tokenizer)
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.use_gumbel_softmax = use_gumbel_softmax

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through the whole cycle in both directions

        1) input -> expander -> intermediate output -> compressor -> reconstructed input
        2) input -> compressor -> intermediate output -> expander -> reconstructed input

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # ==============================================
        # ==============================================
        # INFO: First Direction (Expander - Compressor)
        # ==============================================
        # ==============================================

        original_input = dict_input.copy()

        # INFO - Step 1: Expansion (Summary -> Generated Story)
        expansion_results = self.expander(dict_input)
        expansion_loss_1, expansion_logits_1, expansion_accuracy_1, expansion_bleu_1 = (
            expansion_results["loss"],
            expansion_results["logits"],
            expansion_results["accuracy"],
            expansion_results["bleu"],
        )

        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(expansion_logits_1, self.compressor.get_embeddings())

            # pass generated story embeddings to compressor
            dict_input["story_embs"] = embs

            del embs
        else:
            generated_story_ids = argmax(expansion_logits_1, dim=-1)

            # overwrite dict_input['story_ids'] (original story ids) with generated_story_ids
            dict_input["story_ids"] = generated_story_ids

            del generated_story_ids

        del expansion_results

        # INFO - Step 2: Compression (Generated Story -> Reconstructed Summary)
        compression_results = self.compressor(dict_input)
        compression_loss_1, compression_logits_1, compression_accuracy_1, compression_bleu_1 = (
            compression_results["loss"],
            compression_results["logits"],
            compression_results["accuracy"],
            compression_results["bleu"],
        )

        del compression_results

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
        compression_results = self.compressor(dict_input)
        compression_loss_2, compression_logits_2, compression_accuracy_2, compression_bleu_2 = (
            compression_results["loss"],
            compression_results["logits"],
            compression_results["accuracy"],
            compression_results["bleu"],
        )

        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(compression_logits_2, self.expander.get_embeddings())

            # pass generated summary embeddings to compressor
            dict_input["summary_embs"] = embs

            del embs
        else:
            generated_summary_ids = argmax(compression_logits_2, dim=-1)

            # overwrite dict_input['summary_ids'] (original summary ids) with generated_summary_ids
            dict_input["summary_ids"] = generated_summary_ids

            del generated_summary_ids

        del compression_results

        # INFO - Step 2: Expansion (Generated Summary -> Reconstructed Story)
        expansion_results = self.expander(dict_input)
        expansion_loss_2, expansion_logits_2, expansion_accuracy_2, expansion_bleu_2 = (
            expansion_results["loss"],
            expansion_results["logits"],
            expansion_results["accuracy"],
            expansion_results["bleu"],
        )

        del expansion_results

        # ==============================================
        # ==============================================
        # INFO - Step 3: Calculate Aggregated Metrics
        # ==============================================
        # ==============================================

        expansion_loss = expansion_loss_1 + expansion_loss_2
        expansion_bleu = mean(tensor([expansion_bleu_1, expansion_bleu_2], device=self.device))
        expansion_accuracy = mean(tensor([expansion_accuracy_1, expansion_accuracy_2], device=self.device))

        compression_loss = compression_loss_1 + compression_loss_2
        compression_bleu = mean(tensor([compression_bleu_1, compression_bleu_2], device=self.device))
        compression_accuracy = mean(tensor([compression_accuracy_1, compression_accuracy_2], device=self.device))

        total_loss = expansion_loss + compression_loss
        aggr_accuracy = mean(tensor([expansion_accuracy, compression_accuracy], device=self.device))
        aggr_bleu = mean(tensor([expansion_bleu, compression_bleu], device=self.device))

        return {
            # losses
            "loss": total_loss,
            "exp_loss": expansion_loss,
            "comp_loss": compression_loss,
            # accuracy
            "acc": aggr_accuracy.detach(),
            "exp_acc": expansion_accuracy.detach(),
            "comp_acc": compression_accuracy.detach(),
            # bleu
            "bleu": aggr_bleu.detach(),
            "exp_bleu": expansion_bleu.detach(),
            "comp_bleu": compression_bleu.detach(),
        }

    def generate(self, conditioning_sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate intermediate stories and reconstructed summaries based on
            conditional input summaries
        """

        generated_stories = self.expander.generate(conditioning_sentences)
        reconstructed_summaries = self.compressor.generate(generated_stories)

        return generated_stories, reconstructed_summaries

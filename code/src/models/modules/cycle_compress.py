from typing import Dict, List, Tuple

from torch import argmax, cuda, device, mean, nn, tensor
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander
from src.utils.model_utils import get_gumbel_sampled_embeddings


class CycleArchitectureCompress(nn.Module):
    def __init__(
        self,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool = False,
    ):
        super().__init__()

        assert expander_model_name == compressor_model_name

        self.tokenizer = BartTokenizerFast.from_pretrained(expander_model_name)
        self.expander = Expander(model_name_or_path=expander_model_name, tokenizer=self.tokenizer)
        self.compressor = Compressor(model_name_or_path=compressor_model_name, tokenizer=self.tokenizer)
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.use_gumbel_softmax = use_gumbel_softmax

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through the whole cycle
        input -> expander -> intermediate output -> compressor -> reconstructed input

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # INFO - Step 1: Expansion (Summary -> Generated Story)
        expansion_results = self.expander(dict_input)
        expansion_loss, expansion_logits, expansion_accuracy, expansion_bleu = (
            expansion_results["loss"],
            expansion_results["logits"],
            expansion_results["accuracy"],
            expansion_results["bleu"],
        )

        # INFO: if using gumbel then the whole cycle is differentiable
        #  if not using gumbel then dual learning technique
        if self.use_gumbel_softmax:
            embs = get_gumbel_sampled_embeddings(expansion_logits, self.compressor.get_embeddings())

            # pass generated story embeddings to compressor
            dict_input["story_embs"] = embs

            del embs
        else:
            generated_story_ids = argmax(expansion_logits, dim=-1)

            # overwrite dict_input['story_ids'] (original story ids) with generated_story_ids
            dict_input["story_ids"] = generated_story_ids

            del generated_story_ids

        # del expansion_results

        # INFO - Step 2: Compression (Generated Story -> Reconstructed Summary)
        compression_results = self.compressor(dict_input)
        compression_loss, compression_logits, compression_accuracy, compression_bleu = (
            compression_results["loss"],
            compression_results["logits"],
            compression_results["accuracy"],
            compression_results["bleu"],
        )

        # INFO - Step 3: Calculate Aggregated Metrics
        total_loss = expansion_loss + compression_loss
        aggr_accuracy = mean(tensor([expansion_accuracy, compression_accuracy], device=self.device))
        aggr_bleu = mean(tensor([expansion_bleu, compression_bleu], device=self.device))

        del compression_results

        return {
            # losses
            "loss": total_loss,
            "exp_loss": expansion_loss,
            "comp_loss": compression_loss,
            # accuracy
            "acc": aggr_accuracy.detach(),
            "exp_acc": expansion_accuracy,
            "comp_acc": compression_accuracy,
            # bleu
            "bleu": aggr_bleu.detach(),
            "exp_bleu": expansion_bleu,
            "comp_bleu": compression_bleu,
            # extra exp metric
            "exp_bleu1": expansion_results["bleu1"],
            "exp_bleu2": expansion_results["bleu2"],
            "exp_bleu3": expansion_results["bleu3"],
            "exp_bleu4": expansion_results["bleu4"],
            "exp_ppl": expansion_results["ppl"],
            "exp_dstnct1": expansion_results["distinct1"],
            "exp_dstnct2": expansion_results["distinct2"],
            "exp_dstnct3": expansion_results["distinct3"],
            "exp_dstnct4": expansion_results["distinct4"],
        }

    def generate(self, conditioning_sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate intermediate stories and reconstructed summaries based on
            conditional input summaries
        """

        generated_stories = self.expander.generate(conditioning_sentences)
        reconstructed_summaries = self.compressor.generate(generated_stories)

        return generated_stories, reconstructed_summaries

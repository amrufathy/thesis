from typing import Dict, List, Tuple

from torch import argmax, cuda, device, mean, nn, tensor
from torch.nn.functional import gumbel_softmax
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander


class CycleArchitectureExpand(nn.Module):
    def __init__(
        self,
        expander_model_name: str,
        compressor_model_name: str,
        use_gumbel_softmax: bool = False,
    ):
        super().__init__()

        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.compressor = Compressor(model_name_or_path=compressor_model_name, tokenizer=self.tokenizer)
        self.expander = Expander(model_name_or_path=expander_model_name, tokenizer=self.tokenizer)
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.use_gumbel_softmax = use_gumbel_softmax

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through the whole cycle
        input -> compressor -> intermediate output -> expander -> reconstructed input

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # INFO - Step 1: Compression (Story -> Generated Summary)
        compression_results = self.compressor(dict_input)
        compression_loss, compression_logits, compression_accuracy, compression_bleu = (
            compression_results["loss"],
            compression_results["logits"],
            compression_results["accuracy"],
            compression_results["bleu"],
        )

        # INFO: if using gumbel then the whole cycle is differentiable
        #  if not using gumbel then dual learning technique
        if self.use_gumbel_softmax:
            # WIP
            dists = gumbel_softmax(compression_logits, dim=-1, hard=True)
            embedding = self.expander.expander.get_input_embeddings().weight
            flat_probs = dists.contiguous().view(-1, dists.size(-1))
            flat_embs = flat_probs.mm(embedding)
            embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))

            # pass generated story embeddings to compressor
            dict_input["summary_embs"] = embs

            del dists, embedding, flat_probs, flat_embs, embs
        else:
            generated_summary_ids = argmax(compression_logits, dim=-1)

            # overwrite dict_input['summary_ids'] (original summary ids) with generated_summary_ids
            dict_input["summary_ids"] = generated_summary_ids

            del generated_summary_ids

        del compression_results

        # INFO - Step 2: Expansion (Generated Summary -> Reconstructed Story)
        expansion_results = self.expander(dict_input)
        expansion_loss, expansion_logits, expansion_accuracy, expansion_bleu = (
            expansion_results["loss"],
            expansion_results["logits"],
            expansion_results["accuracy"],
            expansion_results["bleu"],
        )

        # INFO - Step 3: Calculate Aggregated Metrics
        total_loss = expansion_loss + compression_loss
        aggr_accuracy = mean(tensor([expansion_accuracy, compression_accuracy], device=self.device))
        aggr_bleu = mean(tensor([expansion_bleu, compression_bleu], device=self.device))

        del expansion_results

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
        }

    def generate(self, conditioning_sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate intermediate summaries and reconstructed stories based on
            conditional input stories
        """

        generated_summaries = self.compressor.generate(conditioning_sentences)
        reconstructed_stories = self.expander.generate(generated_summaries)

        return reconstructed_stories, generated_summaries

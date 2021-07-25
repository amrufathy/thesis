from typing import Dict, List, Tuple

from torch import argmax, cuda, device, mean, nn, tensor
from torch.nn.functional import gumbel_softmax
from transformers import BartTokenizerFast

from src.models.modules import Compressor, Expander


class CycleArchitectureCompress(nn.Module):
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
            # WIP
            # https://pytorch.org/docs/stable/nn.functional.html#gumbel-softmax
            # https://github.com/cbaziotis/seq3/blob/master/modules/modules.py#L515
            # https://github.com/huggingface/transformers/issues/7693
            # https://stackoverflow.com/questions/61567599/huggingface-bert-inputs-embeds-giving-unexpected-result

            dists = gumbel_softmax(expansion_logits, dim=-1, hard=True)
            embedding = self.compressor.compressor.get_input_embeddings().weight
            flat_probs = dists.contiguous().view(-1, dists.size(-1))
            flat_embs = flat_probs.mm(embedding)
            embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))

            # pass generated story embeddings to compressor
            dict_input["story_embs"] = embs

            del dists, embedding, flat_probs, flat_embs, embs
        else:
            generated_story_ids = argmax(expansion_logits, dim=-1)

            # overwrite dict_input['story_ids'] (original story ids) with generated_story_ids
            dict_input["story_ids"] = generated_story_ids

            del generated_story_ids

        del expansion_results

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
        }

    def generate(self, conditioning_sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate intermediate stories and reconstructed summaries based on
            conditional input summaries
        """

        generated_stories = self.expander.generate(conditioning_sentences)
        reconstructed_summaries = self.compressor.generate(generated_stories)

        return generated_stories, reconstructed_summaries

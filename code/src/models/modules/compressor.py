from torch import nn, argmax, tensor
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right
from typing import Dict, List
from torchmetrics.functional import accuracy
from sacrebleu import corpus_bleu


class Compressor(nn.Module):
    def __init__(self, tokenizer: BartTokenizerFast = None, *args, **kwargs):
        super().__init__()

        self.compressor: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base')

        if not tokenizer:
            self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
        else:
            self.tokenizer = tokenizer

    def forward(self, dict_input: Dict) -> Dict:
        """
        runs the input through compressor bart
        input story -> compressor -> generated summary

        @param dict_input: contains input_ids, attention_masks, labels for both story and summary
        """

        # shift `decoder ids` & `mask` to the right
        summary_ids_shifted = shift_tokens_right(dict_input['summary_ids'], self.tokenizer.pad_token_id,
                                                 self.tokenizer.eos_token_id)
        # summary_msk_shifted = shift_tokens_right(dict_input['summary_attn_msk'], 0, 1)

        # feed the model
        compression_results = self.compressor(**{
            'input_ids': dict_input['story_ids'],
            'attention_mask': dict_input['story_attn_msk'],  # TODO: check if right in case of used within cycle
            'decoder_input_ids': summary_ids_shifted,
            'decoder_attention_mask': dict_input['summary_attn_msk'],
            'labels': dict_input['summary_labels']
        }, use_cache=False)

        compression_loss, compression_logits = compression_results.loss, compression_results.logits

        del summary_ids_shifted, compression_results

        # compute metrics

        # accuracy
        generated_summary_ids = argmax(compression_logits, dim=-1)
        masked_labels = dict_input['summary_labels'].detach().clone()
        masked_labels[masked_labels[:, :] == -100] = self.tokenizer.pad_token_id  # restore padding token id
        acc = accuracy(generated_summary_ids, masked_labels)

        # bleu
        predictions = self.tokenizer.batch_decode(generated_summary_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(masked_labels, skip_special_tokens=True)
        # predictions = self.adjust_padding(predictions, references)

        bleu = corpus_bleu(predictions, [references])

        del generated_summary_ids, masked_labels, predictions, references

        return {'loss': compression_loss, 'logits': compression_logits,
                'accuracy': acc, 'bleu': tensor(bleu.score, device='cuda')}

    def generate(self, conditioning_sentences: List[str]) -> List[str]:
        """
        Generate summaries depending on conditional input stories
        """

        tokenized_sentences = self.tokenizer(conditioning_sentences, padding='longest', return_tensors='pt')

        generated_ids = self.compressor.generate(**tokenized_sentences)

        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_summaries

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
                ts = tp[:len(tr)]
                adjusted_paddings.append(self.tokenizer.convert_tokens_to_string(ts))
            else:  # len(tp) < len(tr)
                # Ġ character represents a whitespace in the byte-level representation
                ts = tp + ['Ġ' + self.tokenizer.pad_token] * (len(tr) - len(tp))
                adjusted_paddings.append(self.tokenizer.convert_tokens_to_string(ts))

            del tp, tr

        return adjusted_paddings

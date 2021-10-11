from typing import Tuple

from torch import Tensor, clone, nn, ones_like, roll
from torch.nn import functional as F
from transformers import (
    BatchEncoding,
    BertConfig,
    BertModel,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
)


class SemanticBert(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model_path = kwargs.get("semantic_bert_path")

        config: BertConfig = BertConfig.from_pretrained(model_path)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_path)
        self.bert: BertModel = BertModel.from_pretrained(model_path)

        # TODO: Uncomment to freeze pretrained params
        # INFO: Leaves out bert.pooler ([CLS] vector) to train
        self.bert.embeddings.requires_grad_(False)
        self.bert.encoder.requires_grad_(False)

        self.linear = nn.Linear(config.hidden_size, 1)

        # https://gombru.github.io/2019/04/03/ranking_loss/
        # https://github.com/nladuo/weak_supervision_ltr/blob/main/code-bert/train_bert.py#L139
        # https://github.com/mikvrax/TrecingLab/blob/master/rank_model_embed.py#L168
        # https://stats.stackexchange.com/questions/332975/training-with-a-max-margin-ranking-loss-converges-to-useless-solution
        self.loss_fn = nn.MarginRankingLoss(margin=1.0, reduction="sum")

    def forward(self, dict_input: BatchEncoding):
        """
        Takes as input the label story and a sample story (generated/negative)
            and outputs a score
        """
        bert_output = self.bert(**dict_input)
        cls = bert_output.pooler_output

        return F.sigmoid(self.linear(cls))

    def prepare_pairs(
        self, gold_stories_ids: Tensor, generated_stories_ids: Tensor, external_tokenizer: PreTrainedTokenizerFast
    ) -> Tuple[BatchEncoding, BatchEncoding]:
        # decoding bart
        generated_stories = external_tokenizer.batch_decode(generated_stories_ids, skip_special_tokens=True)
        gold_stories = external_tokenizer.batch_decode(gold_stories_ids, skip_special_tokens=True)
        false_stories = external_tokenizer.batch_decode(roll(gold_stories_ids, 1, 0), skip_special_tokens=True)

        # break in graph here

        # encode bert
        gold_pos_pairs = self.tokenizer(
            gold_stories,
            generated_stories,
            padding=True,
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        ).to(self.bert.device)

        gold_neg_pairs = self.tokenizer(
            gold_stories, false_stories, padding=True, truncation=True, return_token_type_ids=True, return_tensors="pt"
        ).to(self.bert.device)

        return gold_pos_pairs, gold_neg_pairs

    def compute_loss(self, gold_pos_pairs: BatchEncoding, gold_neg_pairs: BatchEncoding) -> Tensor:
        gold_pos_scores = self.forward(gold_pos_pairs)
        gold_neg_scores = self.forward(gold_neg_pairs)
        targets = ones_like(gold_pos_scores)

        return self.loss_fn(gold_pos_scores, gold_neg_scores, targets)

    def step(
        self, gold_stories_ids: Tensor, generated_stories_ids: Tensor, external_tokenizer: PreTrainedTokenizerFast
    ) -> Tensor:
        gold_pos_pairs, gold_neg_pairs = self.prepare_pairs(
            gold_stories_ids=clone(gold_stories_ids),
            generated_stories_ids=clone(generated_stories_ids),
            external_tokenizer=external_tokenizer,
        )

        return self.compute_loss(gold_pos_pairs, gold_neg_pairs)

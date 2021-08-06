from typing import List


def get_gumbel_sampled_embeddings(logits, transformer_embeddings):
    """
    Obtain embeddings from a differentiable gumbel softmax approximation of input logits
    """
    # https://pytorch.org/docs/stable/nn.functional.html#gumbel-softmax
    # https://github.com/cbaziotis/seq3/blob/master/modules/modules.py#L515
    # https://github.com/huggingface/transformers/issues/7693
    # https://stackoverflow.com/questions/61567599/huggingface-bert-inputs-embeds-giving-unexpected-result

    from torch.nn.functional import gumbel_softmax

    dists = gumbel_softmax(logits, dim=-1, hard=True)
    flat_probs = dists.contiguous().view(-1, dists.size(-1))
    flat_embs = flat_probs.mm(transformer_embeddings)
    embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))

    del dists, flat_probs, flat_embs

    return embs


def distinct_n(candidates_list: List[str], ngram_size: int = 2):
    """
    Distinct is an algorithm for evaluating the textual diversity of the generated
    text by calculating the number of distinct n-grams. See details at https://arxiv.org/abs/1510.03055.

    Code modified from:
    https://paddlenlp.readthedocs.io/en/latest/source/paddlenlp.metrics.distinct.html
    """

    from nltk import ngrams

    diff_ngrams = set()
    count = 0

    for candidate in candidates_list:
        all_ngrams = list(ngrams(candidate.split(), ngram_size))
        diff_ngrams.update(set(all_ngrams))
        count += len(all_ngrams)

    if count == 0:  # prevent division by zero
        return 0

    return len(diff_ngrams) / count


if __name__ == "__main__":
    cand = "The cat The cat on the mat"
    print(distinct_n([cand], 1))  # 0.714
    print(distinct_n([cand], 2))  # 0.833
    print(distinct_n([cand], 3))  # 1
    print(distinct_n([cand], 4))  # 1

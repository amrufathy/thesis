from typing import Dict, List

# https://github.com/thu-coai/HINT/blob/main/eval/eval.py
# https://github.com/thu-coai/Stylized-Story-Generation-with-Style-Guided-Planning/blob/main/analyze/bleu/bleu.py


def distinct_n(candidates_list: List[str], NGRAM_ORDER: int = 4, prefix: str = "") -> Dict:
    """
    Computes the percentage of unique n-grams in a corpus. See details at https://arxiv.org/abs/1510.03055.

    Code modified from:
    https://paddlenlp.readthedocs.io/en/latest/source/paddlenlp.metrics.distinct.html
    """

    from nltk import ngrams, word_tokenize

    distinct_precisions = dict()

    candidates_list = [word_tokenize(c) for c in candidates_list]

    for i in range(1, NGRAM_ORDER + 1):
        diff_ngrams = set()
        count = 0

        for candidate in candidates_list:
            all_ngrams = list(ngrams(candidate, i))
            diff_ngrams.update(set(all_ngrams))
            count += len(all_ngrams)

        if count == 0:  # prevent division by zero
            distinct_precisions[f"{prefix}dstnct{i}"] = 0.0
        else:
            distinct_precisions[f"{prefix}dstnct{i}"] = len(diff_ngrams) / float(count)

    return distinct_precisions


def bleu(ref_streams: List[str], sys_streams: List[str], NGRAM_ORDER: int = 4, prefix: str = "") -> Dict:
    """
    Computes BLEU score for a corpus using NLTK's implementation.

    Returns precisions for each n-gram order
    Returns aggregated score similar to SacreBLEU
    """
    import numpy as np
    from nltk import word_tokenize
    from nltk.translate.bleu_score import corpus_bleu

    sys = [word_tokenize(p) for p in sys_streams]
    refs = [[word_tokenize(r)] for r in ref_streams]

    bleu_precisions = dict()

    for i in range(1, NGRAM_ORDER + 1):
        bleu_precisions[f"{prefix}bleu{i}"] = 100 * corpus_bleu(refs, sys, weights=([1.0 / i for _ in range(i)]))

    # https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py#L278
    bleu_aggr = np.exp(np.sum(np.log(list(bleu_precisions.values()))) / NGRAM_ORDER)

    bleu_precisions[f"{prefix}bleu"] = bleu_aggr

    return bleu_precisions


if __name__ == "__main__":
    cand = "The cat The cat on the mat"
    # D-1 = 0.714, D-2 = 0.833, D-3 = 1, D-4 = 1
    print(distinct_n([cand]))

from torch.nn.functional import gumbel_softmax


def get_gumbel_sampled_embeddings(logits, transformer_embeddings):
    # https://pytorch.org/docs/stable/nn.functional.html#gumbel-softmax
    # https://github.com/cbaziotis/seq3/blob/master/modules/modules.py#L515
    # https://github.com/huggingface/transformers/issues/7693
    # https://stackoverflow.com/questions/61567599/huggingface-bert-inputs-embeds-giving-unexpected-result

    dists = gumbel_softmax(logits, dim=-1, hard=True)
    flat_probs = dists.contiguous().view(-1, dists.size(-1))
    flat_embs = flat_probs.mm(transformer_embeddings)
    embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))

    del dists, flat_probs, flat_embs

    return embs

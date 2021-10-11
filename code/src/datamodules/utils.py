import re

from nltk.tokenize import RegexpTokenizer

"""
Utils for Writing Prompts
"""

REMOVE_CHARS = r"""#$%&()*+/-'"<=>@[\]^_`{|}~"""


class MyRegexpTokenizer(RegexpTokenizer):
    def __call__(self, text: str):
        return self.tokenize(text)


# https://github.com/fangleai/Outline2Story/blob/master/data/util.py#L118
def clean(text: str) -> str:
    text = (
        text.replace("<newline>", "\n").replace("’ ", "'").replace(" n't", "n't").replace("n ’ t", "n't")
    )  # standardize
    text = re.sub(r"\[(.){4}\]", "", text)
    text = re.sub(r"“|``|''|”|‘", '"', text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)  # replace multiple consecutive chars
    text = text.translate(str.maketrans("", "", REMOVE_CHARS))  # remove all except dot
    text = " ".join(text.split())

    return text.strip()


def clean_and_limit_length(text: str) -> str:
    """
    - Clean text
    - Limit target to ten sentences
    """
    from nltk import sent_tokenize

    text_sent = sent_tokenize(text)[:10]
    text = clean(" ".join(text_sent))

    return text

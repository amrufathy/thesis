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
def clean(text: str, remove_first: bool = False) -> str:
    text = (
        text.replace("<newline>", "\n").replace("’ ", "'").replace(" n't", "n't").replace("n ’ t", "n't")
    )  # standardize
    text = re.sub(r"“|``|''|”|‘", '"', text)
    # text = re.sub(r"[.]{2,}", ".", text)  # replace multiple dots
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)  # replace multiple consecutive chars
    text = text.translate(str.maketrans("", "", REMOVE_CHARS))  # remove all except dot
    text_tokens = text.split()[1:] if remove_first else text.split()
    text = " ".join(text_tokens)

    return text.strip()

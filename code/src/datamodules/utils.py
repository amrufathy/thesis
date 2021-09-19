from nltk.tokenize import RegexpTokenizer

"""
Utils for Writing Prompts
"""


class MyRegexpTokenizer(RegexpTokenizer):
    def __call__(self, text: str):
        return self.tokenize(text)


def clean(text: str) -> str:
    t1 = MyRegexpTokenizer(pattern=r"\bnewline\b", gaps=True)
    t2 = MyRegexpTokenizer(pattern=r"\w+|\.")

    intermediate = " ".join(t1(text))  # removes the weird `newline` token

    return " ".join(t2(intermediate)[1:])  # skip the first token

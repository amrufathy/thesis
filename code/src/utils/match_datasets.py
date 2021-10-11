import pandas as pd

# load raw roc
df1 = pd.read_csv("../../data/ROCStories_spring2016.csv")
df2 = pd.read_csv("../../data/ROCStories_winter2017.csv")

raw_roc = pd.concat([df1, df2], ignore_index=True)
raw_roc = raw_roc.drop("storyid", axis=1)


def clean(story: str) -> str:
    story = story.replace("\\", "")
    return story


wanted_summary_keys = ["sentence1"]
wanted_story_keys = ["sentence2", "sentence3", "sentence4", "sentence5"]
raw_roc["summary"] = raw_roc["sentence1"]
raw_roc["story"] = raw_roc.loc[:, wanted_story_keys].apply(lambda x: " ".join(x), axis=1)
raw_roc["summary"] = raw_roc["summary"].apply(lambda x: clean(x))
raw_roc["story"] = raw_roc["story"].apply(lambda x: clean(x))


# load masked HINT
hint_train_df = pd.read_csv("../../HINT_Data/Data/ini_data/roc/train.source", sep="\n", header=None)
hint_val_df = pd.read_csv("../../HINT_Data/Data/ini_data/roc/val.source", sep="\n", header=None)
hint_test_df = pd.read_csv("../../HINT_Data/Data/ini_data/roc/test.source", sep="\n", header=None)

comp_train_df = raw_roc.iloc[: len(hint_train_df)]
comp_val_df = raw_roc.iloc[len(hint_train_df) : len(hint_train_df) + len(hint_val_df)]
comp_test_df = raw_roc.iloc[len(hint_train_df) + len(hint_val_df) :]

comp_train_df.to_csv("../data/roc/roc_hint_raw_train.csv", index=False)
comp_val_df.to_csv("../data/roc/roc_hint_raw_val.csv", index=False)
comp_test_df.to_csv("../data/roc/roc_hint_raw_test.csv", index=False)

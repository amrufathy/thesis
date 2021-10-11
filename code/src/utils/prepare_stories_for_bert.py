import pandas as pd

df1 = pd.read_csv("../../data/ROCStories_spring2016.csv")
df2 = pd.read_csv("../../data/ROCStories_winter2017.csv")

df = pd.concat([df1, df2])

wanted_keys = ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
df["story"] = df.loc[:, wanted_keys].apply(lambda x: " ".join(x), axis=1)

df = df.drop(["storyid", "storytitle", *wanted_keys], axis=1)

df.to_csv("../../data/roc_stories_lines_bert.txt", sep="\n", header=False, index=False)

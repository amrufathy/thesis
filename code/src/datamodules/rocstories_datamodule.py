from os.path import join
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast

"""
ROC dataset

Statistics about story title length:
(All numbers after tokenization)
Max: 33, Mean: 5.05, Std Dev: 1.32, Median: 5.00,
95 percentile: 7.0, 99 percentile: 9.0

Statistics about story length
(All numbers after tokenization)
Max: 110, Mean: 53.35, Std Dev: 9.73, Median: 53.00,
95 percentile: 70.0, 99 percentile: 75.0
"""


class ROCStoriesDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        processed_data_dir: str,
        train_files: Union[str, List[str]],
        batch_size: int = 32,
        percentage: int = 100,
        max_story_length: int = 150,
        max_summary_length: int = 10,
        shuffle: bool = True,
        truncation: Union[str, bool] = True,
        padding: Union[str, bool] = "max_length",
        load_dataset_from_file: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.percentage = percentage
        self.shuffle = shuffle
        self.padding = padding
        self.truncation = truncation
        self.max_story_length = max_story_length
        self.max_summary_length = max_summary_length

        self.load_from_file = load_dataset_from_file

        self.processed_dataset_path = processed_data_dir
        self.train_paths = [join(data_dir, path) for path in train_files]

        self.tokenized_dataset = None
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    def prepare_data(self):
        pass

    def tokenize_example(self, example: Dict):
        example = pd.DataFrame(example)
        wanted_keys = ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
        example["story"] = example.loc[:, wanted_keys].apply(lambda x: " ".join(x), axis=1)
        example = example.to_dict(orient="list")

        story_embeddings = self.tokenizer(
            example["story"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_story_length,
        )
        summary_embeddings = self.tokenizer(
            example["storytitle"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_summary_length,
        )

        story_labels = torch.tensor(story_embeddings["input_ids"])
        summary_labels = torch.tensor(summary_embeddings["input_ids"])

        story_labels[story_labels[:, :] == self.tokenizer.pad_token_id] = -100
        summary_labels[summary_labels[:, :] == self.tokenizer.pad_token_id] = -100

        return {
            "story_ids": story_embeddings["input_ids"],
            "story_attn_msk": story_embeddings["attention_mask"],
            "story_labels": story_labels.tolist(),
            "summary_ids": summary_embeddings["input_ids"],
            "summary_attn_msk": summary_embeddings["attention_mask"],
            "summary_labels": summary_labels.tolist(),
        }

    def setup(self, *args, **kwargs):
        if not self.load_from_file:
            # load dataset from csv file
            dataset = load_dataset(
                "csv",
                data_files={"train": self.train_paths},
                split=f"train[:{self.percentage}%]",
            )

            # self.stats(dataset)

            # train/val/test split -> 80/10/10
            train_test_data = dataset.train_test_split(test_size=0.2, seed=42)
            test_val_data = train_test_data["test"].train_test_split(test_size=0.5, seed=42)
            dataset = DatasetDict(
                {
                    "train": train_test_data["train"],
                    "val": test_val_data["train"],
                    "test": test_val_data["test"],
                }
            )

            # tokenize data
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True)
            self.tokenized_dataset.remove_columns_(["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"])

            self.tokenized_dataset.save_to_disk(self.processed_dataset_path)

            del dataset, train_test_data, test_val_data
        else:
            self.tokenized_dataset = load_from_disk(self.processed_dataset_path)

        # pytorch vector format
        self.tokenized_dataset.set_format(
            type="torch",
            columns=[
                "story_ids",
                "story_attn_msk",
                "story_labels",
                "summary_ids",
                "summary_attn_msk",
                "summary_labels",
            ],
        )

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["train"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["val"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_dataset["test"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def stats(self, dataset):
        def length_stats(lst):
            lengths = [len(i) for i in lst]
            print(
                f"Max: {np.max(lengths)}, Mean: {np.mean(lengths):.2f}, Std Dev: {np.std(lengths):.2f}, "
                f"Median: {np.median(lengths):.2f}, 95 percentile: {np.percentile(lengths, 95)}, "
                f"99 percentile: {np.percentile(lengths, 99)}"
            )

        # check
        print(dataset)

        # stats for story titles/summaries
        titles = dataset["storytitle"]
        tok_titles = self.tokenizer(titles)
        length_stats(list(tok_titles.values())[0])

        # stats for stories
        dataset = pd.DataFrame(dataset)
        wanted_keys = ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
        dataset["story"] = dataset.loc[:, wanted_keys].apply(lambda x: " ".join(x), axis=1)
        dataset = dataset.to_dict(orient="list")

        stories = dataset["story"]
        tok_stories = self.tokenizer(stories)
        length_stats(list(tok_stories.values())[0])

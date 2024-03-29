from os.path import join
from typing import Dict, List, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast

"""
ROC dataset

Statistics about story title length:
(BEFORE tokenization)
Max: 25, Mean: 2.24, Std Dev: 1.09, Median: 2.00,
95 percentile: 4.0, 99 percentile: 5.0

(AFTER tokenization)
Max: 33, Mean: 5.05, Std Dev: 1.32, Median: 5.00,
95 percentile: 7.0, 99 percentile: 9.0


Statistics about story length
(BEFORE tokenization)
Max: 76, Mean: 44.10, Std Dev: 8.92, Median: 44.00,
95 percentile: 59.0, 99 percentile: 64.0

(AFTER tokenization)
Max: 110, Mean: 53.35, Std Dev: 9.73, Median: 53.00,
95 percentile: 70.0, 99 percentile: 75.0
"""

"""
ROC dataset (One Sentence)

Statistics about summary length:
(BEFORE tokenization)
Max: 40, Mean: 10.21, Std Dev: 2.89, Median: 10.00,
95 percentile: 15.0, 99 percentile: 17.0

(AFTER tokenization)
Max: 51, Mean: 14.39, Std Dev: 3.15, Median: 14.00,
95 percentile: 20.0, 99 percentile: 22.0


Statistics about story length
(BEFORE tokenization)
Max: 63, Mean: 36.13, Std Dev: 7.57, Median: 36.00,
95 percentile: 48.0, 99 percentile: 53.0

(AFTER tokenization)
Max: 90, Mean: 43.78, Std Dev: 8.25, Median: 44.00,
95 percentile: 57.0, 99 percentile: 62.0
"""


class ROCStoriesDataModule(LightningDataModule):
    """
    Loads ROC Stories Dataset

    Summary is the `title`
    Story is the `5 sentences concatenation`
    """

    def __init__(
        self,
        data_dir: str,
        processed_data_dir: str,
        train_files: Union[str, List[str]],
        batch_size: int = 32,
        percentage: int = 100,
        max_story_length: int = 70,
        max_summary_length: int = 7,
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
        self.train_paths = join(data_dir, train_files)

        self.tokenized_dataset = None
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    def prepare_data(self):
        pass

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
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True, desc="Tokenizing")
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
                "summary_ids",
                "summary_attn_msk",
            ],
        )

    def tokenize_example(self, example: Dict):
        story_embeddings = self.tokenizer(
            example["story"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_story_length,
        )
        summary_embeddings = self.tokenizer(
            example["summary"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_summary_length,
        )

        return {
            "story_ids": story_embeddings["input_ids"],
            "story_attn_msk": story_embeddings["attention_mask"],
            "summary_ids": summary_embeddings["input_ids"],
            "summary_attn_msk": summary_embeddings["attention_mask"],
        }

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
        from nltk.tokenize import RegexpTokenizer

        tokenizer = RegexpTokenizer(pattern=r"\w+")

        def length_stats(lst):
            lengths = [len(i) for i in lst]
            print(
                f"Max: {np.max(lengths)}, Mean: {np.mean(lengths):.2f}, Std Dev: {np.std(lengths):.2f}, "
                f"Median: {np.median(lengths):.2f}, 95 percentile: {np.percentile(lengths, 95)}, "
                f"99 percentile: {np.percentile(lengths, 99)}"
            )

        # check
        print(dataset)

        # stats for titles/summaries [prompts]
        titles = dataset["storytitle"]
        tok_titles = [tokenizer.tokenize(t) for t in titles]
        length_stats(tok_titles)

        tok_titles = self.tokenizer(titles)
        length_stats(list(tok_titles.values())[0])

        # stats for stories
        stories = dataset["story"]
        tok_stories = [tokenizer.tokenize(s) for s in stories]
        length_stats(tok_stories)

        tok_stories = self.tokenizer(stories)
        length_stats(list(tok_stories.values())[0])

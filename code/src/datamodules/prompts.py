from os.path import join
from typing import Dict, List, Union

import numpy as np
from datasets import ReadInstruction, concatenate_datasets, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast

from src.datamodules.utils import MyRegexpTokenizer

"""
Writing Prompts dataset
Experimental Setup #3

Default split: 90/5/5

[Statistics: All dataset 300K]
Statistics about story title length:
(BEFORE tokenization)
Max: 64, Mean: 22.44, Std Dev: 12.67, Median: 21.00,
95 percentile: 47.0, 99 percentile: 54.0

(AFTER tokenization)
Max: 89, Mean: 28.22, Std Dev: 14.56, Median: 26.00,
95 percentile: 57.0, 99 percentile: 65.0


Statistics about story length:
(BEFORE tokenization)
Max: 6936, Mean: 542.01, Std Dev: 368.24, Median: 447.00,
95 percentile: 1310.0, 99 percentile: 1742.0

(AFTER tokenization)
Token indices sequence length is longer than the specified maximum sequence length for this model (1307 > 1024).
    Running this sequence through the model will result in indexing errors
Max: 17917, Mean: 659.14, Std Dev: 448.02, Median: 543.00,
95 percentile: 1588.0, 99 percentile: 2110.0
"""

"""
Writing Prompts dataset
Experimental Setup #2

Default split: 90/5/5

[Statistics: All dataset 300K]
Statistics about story title length:
(BEFORE tokenization)
Max: 64, Mean: 22.44, Std Dev: 12.67, Median: 21.00,
95 percentile: 47.0, 99 percentile: 54.0

(AFTER tokenization)
Max: 89, Mean: 28.22, Std Dev: 14.56, Median: 26.00,
95 percentile: 57.0, 99 percentile: 65.0


Statistics about story length:
(BEFORE tokenization)
Max: 2433, Mean: 135.00, Std Dev: 53.01, Median: 127.00,
95 percentile: 226.0, 99 percentile: 297.0

(AFTER tokenization)
Token indices sequence length is longer than the specified maximum sequence length for this model (1304 > 1024).
Running this sequence through the model will result in indexing errors
Max: 17917, Mean: 164.85, Std Dev: 72.50, Median: 156.00,
95 percentile: 270.0, 99 percentile: 355.0
"""


class WritingPromptsDataModule(LightningDataModule):
    """
    Loads Writing Prompts Dataset

    Summary is the `prompt`
    Story is the `target paragraphs`
    """

    def __init__(
        self,
        data_dir: str,
        processed_data_dir: str,
        train_files: Union[str, List[str]],
        val_files: Union[str, List[str]],
        test_files: Union[str, List[str]],
        batch_size: int = 32,
        percentage: int = 100,
        max_story_length: int = 1024,
        max_summary_length: int = 65,
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
        self.val_paths = join(data_dir, val_files)
        self.test_paths = join(data_dir, test_files)

        self.tokenized_dataset = None
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    def prepare_data(self):
        pass

    def setup(self, *args, **kwargs):
        if not self.load_from_file:
            # load dataset from text files
            dataset = load_dataset(
                "csv",
                data_files={"train": self.train_paths, "val": self.val_paths, "test": self.test_paths},
                split={
                    "train": ReadInstruction("train", to=self.percentage, unit="%"),
                    "val": ReadInstruction("val", to=self.percentage, unit="%"),
                    "test": ReadInstruction("test", to=self.percentage, unit="%"),
                },
            )

            # self.stats(dataset)

            # tokenize data
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True, desc="Tokenizing")
            self.tokenized_dataset.remove_columns_(["source", "target"])

            self.tokenized_dataset.save_to_disk(self.processed_dataset_path)

            del dataset
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
            example["target"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_story_length,
        )
        summary_embeddings = self.tokenizer(
            example["source"],
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
        tokenizer = MyRegexpTokenizer(pattern=r"\w+")

        def length_stats(lst):
            lengths = [len(i) for i in lst]
            print(
                f"Max: {np.max(lengths)}, Mean: {np.mean(lengths):.2f}, Std Dev: {np.std(lengths):.2f}, "
                f"Median: {np.median(lengths):.2f}, 95 percentile: {np.percentile(lengths, 95)}, "
                f"99 percentile: {np.percentile(lengths, 99)}"
            )

        # check
        print(dataset)
        dataset = concatenate_datasets([dataset["train"], dataset["val"], dataset["test"]])
        print(dataset)

        # stats for titles/summaries [prompts]
        titles = dataset["source"]
        # tok_titles = [clean(t).split() for t in titles]
        tok_titles = [tokenizer(t) for t in titles]
        length_stats(tok_titles)

        tok_titles = self.tokenizer(titles)
        length_stats(list(tok_titles.values())[0])

        # stats for stories
        stories = dataset["target"]
        # tok_stories = [clean(s).split() for s in stories]
        tok_stories = [tokenizer(s) for s in stories]
        length_stats(tok_stories)

        tok_stories = self.tokenizer(stories)
        length_stats(list(tok_stories.values())[0])

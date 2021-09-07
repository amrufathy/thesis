from os.path import join
from typing import Dict, List, Union

import numpy as np
import torch
from datasets import DatasetDict, Features, concatenate_datasets, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast

"""
Writing Prompts dataset

Statistics about story title length:
(All numbers after tokenization)
Max: 99, Mean: 32.03, Std Dev: 14.99, Median: 30.00,
95 percentile: 61.0, 99 percentile: 70.0
"""


class WritingPromptsDataModule(LightningDataModule):
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
        val_files: Union[str, List[str]],
        test_files: Union[str, List[str]],
        batch_size: int = 32,
        percentage: int = 100,
        max_story_length: int = 150,
        max_summary_length: int = 50,
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
        self.val_paths = [join(data_dir, path) for path in val_files]
        self.test_paths = [join(data_dir, path) for path in test_files]

        self.tokenized_dataset = None
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    def prepare_data(self):
        pass

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

    def setup(self, *args, **kwargs):
        if not self.load_from_file:
            # load dataset from text files
            train_dataset = self.load_concatenated_dataset(self.train_paths)
            val_dataset = self.load_concatenated_dataset(self.val_paths)
            test_dataset = self.load_concatenated_dataset(self.test_paths)

            dataset = DatasetDict(
                {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset,
                }
            )

            # tokenize data
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True)
            self.tokenized_dataset.remove_columns_(["source", "target"])

            self.tokenized_dataset.save_to_disk(self.processed_dataset_path)

            del dataset, train_dataset, val_dataset, test_dataset
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

    def load_concatenated_dataset(self, dataset_paths):
        src_path, trgt_path = dataset_paths

        src_dataset = load_dataset(
            "text",
            data_files={"_": src_path},
            split=f"_[:{self.percentage}%]",
            features=Features.from_dict({"source": {"dtype": "string", "_type": "Value"}}),
        )

        trgt_dataset = load_dataset(
            "text",
            data_files={"_": trgt_path},
            split=f"_[:{self.percentage}%]",
            features=Features.from_dict({"target": {"dtype": "string", "_type": "Value"}}),
        )

        dataset = concatenate_datasets([src_dataset, trgt_dataset], axis=1)

        del src_dataset, trgt_dataset

        return dataset

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
                f"Max: {np.max(lengths)}, Mean: {np.mean(lengths):.2f}, Std Dev: {np.std(lengths):.2f},"
                f"Median: {np.median(lengths):.2f}, 95 percentile: {np.percentile(lengths, 95)},"
                f"99 percentile: {np.percentile(lengths, 99)}"
            )

        # check
        print(dataset)

        # stats for story titles/summaries
        titles = dataset["source"]
        tok_titles = self.tokenizer(titles)
        length_stats(list(tok_titles.values())[0])
        length_stats(list(titles))

        # stats for stories
        # stories = dataset["target"]
        # tok_stories = self.tokenizer(stories)
        # length_stats(list(stories.values())[0])
        # length_stats(list(stories))

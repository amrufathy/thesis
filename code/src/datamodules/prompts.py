from os.path import join
from typing import Dict, List, Union

import numpy as np
from datasets import DatasetDict, Features, concatenate_datasets, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast

from src.datamodules.utils import clean

"""
Writing Prompts dataset

Default split: 90/5/5

Statistics about story title length:
(BEFORE tokenization)
Max: 67, Mean: 23.63, Std Dev: 12.76, Median: 22.00,
95 percentile: 48.0, 99 percentile: 56.0

(AFTER tokenization)
Max: 99, Mean: 32.05, Std Dev: 15.00, Median: 30.00,
95 percentile: 61.0, 99 percentile: 70.0


Statistics about story length:
(BEFORE tokenization)
Max: 7323, Mean: 578.92, Std Dev: 390.30, Median: 478.00,
95 percentile: 1392.0, 99 percentile: 1849.0

(AFTER tokenization)
Token indices sequence length is longer than the specified maximum sequence length for this model (1501 > 1024).
    Running this sequence through the model will result in indexing errors.
Max: 18043, Mean: 822.31, Std Dev: 561.40, Median: 675.00,
95 percentile: 1982.0, 99 percentile: 2631.0
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

            # self.stats(dataset)

            # tokenize data
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True, desc="Tokenizing")
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

    def tokenize_example(self, example: Dict):
        example["target"] = [clean(ex) for ex in example["target"]]
        example["source"] = [clean(ex) for ex in example["source"]]

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
        tok_titles = [clean(t).split() for t in titles]
        length_stats(tok_titles)

        tok_titles = self.tokenizer(titles)
        length_stats(list(tok_titles.values())[0])

        # stats for stories
        stories = dataset["target"]
        tok_stories = [clean(s).split() for s in stories]
        length_stats(tok_stories)

        tok_stories = self.tokenizer(stories)
        length_stats(list(tok_stories.values())[0])

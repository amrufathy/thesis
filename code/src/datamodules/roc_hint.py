from os.path import join
from typing import Dict, List, Union

from datasets import ReadInstruction, load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast


class ROCStoriesHINTDataModule(LightningDataModule):
    """
    Loads ROC Stories Dataset matched to HINT paper split

    Summary is the `1st sentence`
    Story is the `remaining 4 sentences concatenation`
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
        shuffle: bool = True,
        load_dataset_from_file: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.percentage = percentage
        self.shuffle = shuffle
        self.max_story_length = 100  # tokens
        self.max_summary_length = 30  # tokens

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
            # load dataset from csv file
            dataset = load_dataset(
                "csv",
                data_files={"train": self.train_paths, "val": self.val_paths, "test": self.test_paths},
                split={
                    "train": ReadInstruction("train", to=self.percentage, unit="%"),
                    "val": ReadInstruction("val", to=self.percentage, unit="%"),
                    "test": ReadInstruction("test", to=self.percentage, unit="%"),
                },
            )

            # tokenize data
            self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True, desc="Tokenizing")
            self.tokenized_dataset.remove_columns_(
                ["storytitle", "sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
            )

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
            example["story"], padding="max_length", truncation=True, max_length=self.max_story_length
        )
        summary_embeddings = self.tokenizer(
            example["summary"], padding="max_length", truncation=True, max_length=self.max_summary_length
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

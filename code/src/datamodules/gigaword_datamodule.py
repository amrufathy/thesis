from os.path import join
from typing import Dict, Optional

import torch
from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BartTokenizerFast


class GigaWordDataModule(LightningDataModule):
    def __init__(
        self, data_dir, batch_size, train_path, val_path, processed_data_dir, **kwargs
    ):
        super().__init__()

        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.tokenized_dataset = None
        self.train_path = join(data_dir, train_path)
        self.validation_path = join(data_dir, val_path)
        self.batch_size = batch_size
        self.processed_dataset_path = join(data_dir, processed_data_dir)

    def tokenize_example(self, example: Dict):
        # https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904
        # https://github.com/huggingface/transformers/issues/7961#issuecomment-714192751
        # shift right is for denoising as in the paper
        # print(len(example['source'][0]), len(example['target'][0]))

        # TODO: configurable params
        article_embeddings = self.tokenizer(
            example["source"], padding="max_length", truncation=True, max_length=50
        )
        summary_embeddings = self.tokenizer(
            example["target"], padding="max_length", truncation=True, max_length=50
        )

        article_labels = torch.tensor(article_embeddings["input_ids"])
        summary_labels = torch.tensor(summary_embeddings["input_ids"])

        article_labels[article_labels[:, :] == self.tokenizer.pad_token_id] = -100
        summary_labels[summary_labels[:, :] == self.tokenizer.pad_token_id] = -100

        return {
            "story_ids": article_embeddings["input_ids"],
            "story_attn_msk": article_embeddings["attention_mask"],
            "story_labels": article_labels.tolist(),
            "summary_ids": summary_embeddings["input_ids"],
            "summary_attn_msk": summary_embeddings["attention_mask"],
            "summary_labels": summary_labels.tolist(),
        }

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # dataset = load_dataset('csv', data_files={'train': self.train_path, 'val': self.validation_path})
        #
        # self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True,
        #                                      # batch_size=int(self.batch_size),
        #                                      cache_file_names={k: f'tokenized_and_grouped_{str(k)}' for k in dataset})
        #
        # self.tokenized_dataset.save_to_disk(self.processed_dataset_path)
        self.tokenized_dataset = load_from_disk(self.processed_dataset_path)

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

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.tokenized_dataset["train"], batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.tokenized_dataset["val"], batch_size=self.batch_size)

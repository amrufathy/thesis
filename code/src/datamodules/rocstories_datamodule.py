from pytorch_lightning import LightningDataModule
from os.path import join
from transformers import BartTokenizerFast
from datasets import load_dataset, load_from_disk, DatasetDict
from typing import Dict
import torch
import pandas as pd
from torch.utils.data.dataloader import DataLoader


class ROCStoriesDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, percentage, shuffle, train_path, processed_data_dir, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.percentage = percentage
        self.shuffle = shuffle
        self.processed_dataset_path = join(data_dir, processed_data_dir)

        self.train_paths = [
            join(data_dir, path) for path in train_path
        ]

        self.tokenized_dataset = None
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

    def prepare_data(self):
        pass

    def tokenize_example(self, example: Dict):
        example = pd.DataFrame(example)
        wanted_keys = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        example['story'] = example.loc[:, wanted_keys].apply(lambda x: ' '.join(x), axis=1)
        example = example.to_dict(orient='list')

        story_embeddings = self.tokenizer(example['story'], padding='max_length', truncation=True, max_length=150)
        summary_embeddings = self.tokenizer(example['storytitle'], padding='max_length', truncation=True, max_length=10)

        story_labels = torch.tensor(story_embeddings['input_ids'])
        summary_labels = torch.tensor(summary_embeddings['input_ids'])

        story_labels[story_labels[:, :] == self.tokenizer.pad_token_id] = -100
        summary_labels[summary_labels[:, :] == self.tokenizer.pad_token_id] = -100

        return {
            'story_ids': story_embeddings['input_ids'],
            'story_attn_msk': story_embeddings['attention_mask'],
            'story_labels': story_labels.tolist(),
            'summary_ids': summary_embeddings['input_ids'],
            'summary_attn_msk': summary_embeddings['attention_mask'],
            'summary_labels': summary_labels.tolist()
        }

    def setup(self, *args, **kwargs):
        # load dataset from csv file
        dataset = load_dataset('csv', data_files={'train': self.train_paths}, split=f'train[:{self.percentage}%]')

        # train/val/test split
        train_test_data = dataset.train_test_split(test_size=0.2)
        test_val_data = train_test_data['test'].train_test_split(test_size=0.5)
        dataset = DatasetDict({
            'train': train_test_data['train'],
            'val': test_val_data['train'],
            'test': test_val_data['test']
        })

        # tokenize data
        self.tokenized_dataset = dataset.map(self.tokenize_example, batched=True)
        # self.tokenized_dataset.remove_columns_(
        #     ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5'])

        # save/load data
        # self.tokenized_dataset.save_to_disk(self.processed_dataset_path)

        # self.tokenized_dataset = load_from_disk(self.processed_dataset_path)

        # pytorch vector format
        self.tokenized_dataset.set_format(type='torch',
                                          columns=['story_ids', 'story_attn_msk', 'story_labels',
                                                   'summary_ids', 'summary_attn_msk', 'summary_labels'])

        del dataset, train_test_data, test_val_data

    def train_dataloader(self):
        return DataLoader(self.tokenized_dataset['train'],
                          batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.tokenized_dataset['val'],
                          batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.tokenized_dataset['test'],
                          batch_size=self.batch_size, shuffle=self.shuffle)

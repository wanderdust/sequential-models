import string

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

"""
Class Generated using ChatGPT
"""


class IMDBDataset(Dataset):
    def __init__(self, split):
        """
        Initializes the dataset by loading the IMDB data and preprocessing it.
        :param split: 'train' or 'test'
        """
        self.data = []
        self.labels = []
        self.char_set = string.ascii_letters + string.digits + string.punctuation + ' '
        self.char_map = {char: i for i, char in enumerate(self.char_set)}
        self.max_len = 100  # Maximum length of a sequence

        imdb_dataset = load_dataset("imdb")
        for sample in imdb_dataset[split]:
            self.data.append(self.one_hot_encode(self.pad(sample['text'])))
            self.labels.append(sample['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def pad(self, text):
        """
        Pads or trims the text to the maximum length.
        """
        return text[:self.max_len].ljust(self.max_len)

    def one_hot_encode(self, text):
        """
        Converts characters to one-hot encoded vectors.
        """
        encoded_text = [self.char_to_vec(char).tolist() for char in text]
        return torch.tensor(encoded_text)

    def char_to_vec(self, char):
        """
        Converts a character to a one-hot encoded vector.
        """
        vec = [0] * len(self.char_set)
        if char in self.char_map:
            vec[self.char_map[char]] = 1
        return torch.tensor(vec)

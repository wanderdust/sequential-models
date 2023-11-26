import random
import string

from torch.utils.data import Dataset


class MySimpleDataset(Dataset):
    """Simple dataset that I use to check if the model is correctly implemented

    If input is a letter, label is 0
    If input is a number, label is 1
    """
    def __init__(self, split, size):
        self.dataset_size = size
        self.data = self._generate_data()

        # Split the data (for simplicity, let's just do a 80-20 split for train-test)
        split_index = int(0.8 * len(self.data))
        if split == 'train':
            self.data = self.data[:split_index]
        elif split == 'test':
            self.data = self.data[split_index:]

    def _generate_data(self):
        dataset = []
        for _ in range(self.dataset_size):
            dataset.append({"text": self._generate_random_string(), "label": 0})
            dataset.append({"text": self._generate_random_number(), "label": 1})
        return dataset

    def _generate_random_string(self, length=1):
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def _generate_random_number(self, length=1):
        return ''.join(random.choices(string.digits, k=length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

"""Dataclass."""
import random

from pathlib import Path
from typing import Union, List, Tuple

from numpy import true_divide

class POSData:
    """Dataclass for tagged parts-of-speech data."""
    def __init__(self,
                 data: Union[str, Path],
                 train: bool,
                 read: bool = True,
                 shuffle: bool = True) -> None:
        self.data = data
        self.train = train
        self.shuffle = shuffle
        self.X, self.y = [], []
        if read:
            self.read_txt()

    def read_txt(self,) -> None:
        # If there is a bug in the future, I swapped 
        # the order of the with() statement with 
        # the if else statements.
        with open(self.data, mode='r') as f:
            for line in f:
                if self.train:
                    cur_X, cur_y = [], []
                    elle = line.split()
                    for word in elle:
                        word_tag = word.split('/')
                        cur_X.append(word_tag[0])
                        cur_y.append(word_tag[1])
                    self.X.append(cur_X)
                    self.y.append(cur_y)
                else:
                    self.X.append(line.split())

    def _shuffle(self,) -> None:
        tmp_list = list(zip(self.X, self.y))
        random.shuffle(tmp_list)
        self.X, self.y = zip(*tmp_list)  # * -> unzip list
    
    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        for x, y in zip(self.X, self.y):
            yield x, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self,):
        return len(self.X)


if __name__ == "__main__":
    data = POSData(
            data="hmm-training-data/it_isdt_train_tagged.txt", 
            train=True, 
            shuffle=True, 
            read=True
        )
    for x, y in data:
        print(x, y)
        break
    

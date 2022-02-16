"""Dataclass."""
from pathlib import Path
from typing import Union

class POSData:
    """Dataclass for tagged parts-of-speech data."""
    def __init__(self,
                 data: Union[str, Path],
                 train: bool,
                 read: bool = True) -> None:
        self.data = data
        self.train = train
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

if __name__ == "__main__":
    data = POSData("hmm-training-data/it_isdt_train_tagged.txt", True, True)
    print(data.X)

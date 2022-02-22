"""Dataclass."""
from pathlib import Path


class POSData:
    """Dataclass for tagged parts-of-speech data."""
    def __init__(self, root, train, read=True):
        self.root = root
        self.train = train
        self.X = None
        self.y = None
        if read:
            self.read_txt()

    def read_txt(self):
        """Note:

        The slash character / is the separator between words and tags, 
        but it also appears within words in the text. Slashes never 
        appear in the tags, so the separator is always the last slash
        in the word/tag sequence. 
        
        i.e., use string.rsplit('/') as opposed to string.split('/').
        """
        with open(self.root, mode='r') as f:
            if self.train:
                self.X, self.y = [], []
                for line in f:
                    cur_X, cur_y = [], []
                    elle = line.split()
                    for word in elle:
                        word_tag = word.rsplit('/', maxsplit=1)
                        assert len(word_tag) == 2
                        assert '/' not in word_tag[1]
                        cur_X.append(word_tag[0])
                        cur_y.append(word_tag[1])
                    self.X.append(cur_X)
                    self.y.append(cur_y)
            else:
                self.X = [line.split() for line in f]
    
    def __iter__(self):
        return zip(self.X, self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        """Returns number of *sequences* in data."""
        return len(self.X)

if __name__ == "__main__":
    print("testing iterator.")
    data = POSData(
            root="hmm-training-data/it_isdt_train_tagged.txt", 
            train=True, 
            read=True
        )
    for x, y in data:
        print(x, y)
        break
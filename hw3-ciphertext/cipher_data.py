import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class CipherNGramData(Dataset):
    def __init__(self, ciphertxtdata, context_size=3, process_now=True):
        self.vocab = None
        self.vocab_size = None
        self.word_to_idx = None
        self.X = None
        self.y = None

        self.text = ciphertxtdata
        self.context_size = context_size
        if process_now:
            self.process_text()

    def process_text(self):
        text = [x for y in self.text for x in y]
        self.vocab = set(text)
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

        self.ngrams = [
            [[text[i - j - 1] for j in range(self.context_size)], text[i]]
            for i in range(self.context_size, len(text))
        ]

        self.X, self.y = [], []
        for i in range(self.context_size, len(text)):
            self.X.append([text[i - j - 1] for j in range(self.context_size)])
            self.y.append(text[i])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        X = self.X[i]
        X = torch.tensor([self.word_to_idx[w] for w in X], dtype=torch.long)
        y = self.y[i]
        y = torch.tensor([self.word_to_idx[y]])

        return X, y


class CipherVecData(Dataset):
    def __init__(self, X, y=None, transform=None, target_transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transfrom = target_transform

    def __len__(self,):
        return len(self.X) if self.X is not None else 0

    def __getitem__(self, i):
        x = self.X[i]
        if self.transform:
            x = self.transform(x)
        if self.target_transfrom and self.y is not None:
            y = self.target_transfrom(self.y[i])
        elif self.y is not None:
            y = self.y[i]

        return x, y if self.y is not None else x


class CipherTxtData(Dataset):
    PATH = {"train": "train_enc.tsv",
            "dev": "dev_enc.tsv",
            "test": "test_enc_unlabeled.tsv"}
    def __init__(self, mode="train", read=True, split=True):
        super().__init__()
        self.data = None
        self.split = split
        try:
            self.root = self.PATH[mode]
            self.mode = mode
        except KeyError:
            raise ValueError(f"{mode} not supported.")
        if read:
            self.read()

    def read(self):
        self.data = []
        with open(self.PATH[self.mode], mode='r', encoding="utf-8") as f:
            for line in f:
                self._read(line, mode=self.mode)
        if self.split:
            self._split()

    def _read(self, line, mode):
        if mode == "test":
           x = line.rstrip('\n\r')
        else:
            x = line.rstrip('\n\r').split('\t')
            x[0] = int(x[0]) # one_hot(int(x[0], num_classes=2)
        self.data.append(x)

    def _split(self):
        if self.mode == "test":
            self.data = [[x.split(' ')] for x in self.data]
        else:
            self.data = [[x[0], x[1].split(' ')] for x in self.data]

    @property
    def X(self):
        return [x[1] for x in self.data] if self.data is not None else None

    @property
    def y(self):
        if self.mode != "test" and self.data is not None:
            return [x[0] for x in self.data]
        else:
            return None

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self,):
        return len(self.data) if self.data is not None else 0


if __name__ == "__main__":
    textdata = CipherTxtData(mode="train")
    newdata = CipherVecData(textdata.X, textdata.y)

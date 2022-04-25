from typing import Callable, Sequence, Union, List, Optional

import numpy as np
import gensim
import torch
from torch.utils.data import Dataset


class CipherTxtData(Dataset):
    """Dataclass for reading cipher-text data from provided files."""
    PATH = {"train": "data/train_enc.tsv",
            "dev": "data/dev_enc.tsv",
            "test": "data/test_enc_unlabeled.tsv"}
    def __init__(self,
                 mode: str = "train",
                 split: bool = True):
        super().__init__()
        self.data: list[list[str]] = []
        self.split: bool = split
        try:
            self.root: str = self.PATH[mode]
            self.mode: str = mode
            self.read()
        except KeyError:
            raise ValueError(f"Mode {mode} not supported.")

    def read(self) -> None:
        """Read datafile."""
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
            x[0] = int(x[0])
        self.data.append(x)

    def _split(self):
        if self.mode == "test":
            self.data = [[x.split(' ')] for x in self.data]
        else:
            self.data = [[x[0], x[1].split(' ')] for x in self.data]

    @property
    def X(self) -> Union[list, None]:
        """Documents."""
        if self.data is None:
            return None
        elif self.mode == "test":
            return self.data
        else:
            return [x[1] for x in self.data] if self.data is not None else None


    @property
    def y(self) -> Union[list, None]:
        """Class labels, either 0 or 1"""
        if self.mode != "test" and self.data is not None:
            return [x[0] for x in self.data]
        else:
            return None

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self,):
        return len(self.data) if self.data is not None else 0


class CipherCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, text):
        self.text = text

    def __iter__(self):
        for line in self.text:
            yield line


class CipherNGramData(Dataset):
    """Dataclass to yield ciphertext ngrams.

    If ngrams are retrieved using `__get_item__` method,
    returns one-hot encoding of ngrams.
    """
    def __init__(self, ciphertxtdata: CipherTxtData, context_size: int = 3):
        self.context_size = context_size
        self.text = [x for y in ciphertxtdata for x in y]
        self.vocab = set(self.text)
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

        self.ngrams = [
            [[self.text[i - j - 1] for j in range(self.context_size)], self.text[i]]
            for i in range(self.context_size, len(self.text))
        ]

        self.X = []
        self.y = []
        for i in range(self.context_size, len(self.text)):
            self.X.append([self.text[i - j - 1] for j in range(self.context_size)])
            self.y.append(self.text[i])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        X = self.X[i]
        X = torch.tensor([self.word_to_idx[w] for w in X], dtype=torch.long)
        y = self.y[i]
        y = torch.tensor([self.word_to_idx[y]])

        return X, y


class CipherW2VData(Dataset):
    """Dataclass that generates w2v embeddings."""
    OOV = '~'
    def __init__(self,
                 corpus: Union[CipherCorpus, list],
                 wv: Optional[gensim.models.Word2Vec] = None,
                 **w2vparams) -> None:
        super().__init__()
        if not isinstance(corpus, CipherCorpus):
            corpus = CipherCorpus(corpus)
        self.corpus = corpus

        if not wv:
            model = gensim.models.Word2Vec(sentences=self.corpus, **w2vparams)
            self.wv = model.wv
            del model
        else:
            self.wv = wv

    def __len__(self):
        return len(self.corpus.text)

    def __getitem__(self, i):
        sentence = self.corpus.text[i]

        sentence_emb = []
        for word in sentence:
            if word in self.wv:
                emb = self.wv[word]
            else:
                emb = self.wv[self.OOV]
            emb = torch.tensor(emb)
            sentence_emb.append(emb)


        return sentence_emb



class CipherVecData(Dataset):
    """General dataclass for ciphertext embeddings."""
    def __init__(self, X: Sequence, y: Optional[Sequence] = None,
                 transform: Callable = None,
                 target_transform: Callable= None) -> None:
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



if __name__ == "__main__":
    textdata = CipherTxtData(mode="train")
    newdata = CipherVecData(textdata.X, textdata.y)

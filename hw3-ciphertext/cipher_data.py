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

    params
    ------
    ciphertxtdata : CipherTxtData
        dataclass containing unprocesses text.
    context_size : int
        "n"-gram size.

    attributes
    ----------
    context_size : int
        "n"-gram size.
    text : list[str]
        flattened list of unprocesses text.
    vocab : set[str]
        all words in self.text.
    vocab_size : int
        number of words in self.vocab.
    word_to_idx : dict
        maps words to index value for one-hot encoding.
    ngrams : list[list[str], str]
        generated ngrams from self.text, where ngrams[i][0] is the
        context , and ngrams[i][1] is the target.
    X : list[str]
        list of all contexts.
    y : list[str]
        list of all targets.



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
    """Dataclass that generates w2v embeddings.

    params
    ------
    corpus : CipherCorpus or list
        iterable of documents that is made to be restartable.
    as_tensor : bool
        whether to convert __getitem__ retrieval to tensor.
    **w2vparams : optional **kwargs
        params to pass to Word2vec model

    attributes
    ----------
    corpus : CipherCorpus
        restartable iterable of documents.
    as_tensor : bool
        whether to convert __getitem__ retrieval to tensor.
    wv : gensim.models.keyedvectors.KeyedVectors
        dictionary mapping words to their w2v embedding
    """

    def __init__(self,
                 corpus: Union[CipherCorpus, list],
                 as_tensor: bool = False,
                 **w2vparams) -> None:
        if not isinstance(corpus, CipherCorpus):
            corpus = CipherCorpus(corpus)
        self.corpus = corpus
        self.as_tensor = as_tensor

        model = gensim.models.Word2Vec(sentences=self.corpus, **w2vparams)
        self.wv = model.wv
        del model

    def __len__(self):
        return len(self.corpus.text)

    def __getitem__(self, i):
        sentence = self.corpus.text[i]
        sentence = [self.wv[word] for word in sentence]
        sentence = np.asarray(sentence)
        if self.as_tensor:
            sentence = torch.from_numpy(sentence)

        return sentence



class CipherVecData(Dataset):
    """General dataclass for ciphertext embeddings.

    params
    ------
    X : list-like
        input data, presumably vectorized.
    y : list-like or None
        target labels
    transform : callable or None
        function to apply to X[i] during retrieval.
    target_transfrom : callable or None
        function to apply to y[i] during retrieval.

    attributes
    ----------
    X : list-list
        input data
    y : list-like or None
        target labels
    transform : callable or None
        function to apply to X[i] during retrieval.
    target_transfrom : callable or None
        function to apply to y[i] during retrieval.
    """
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

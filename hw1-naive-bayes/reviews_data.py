import re
import heapq
from pathlib import Path
from collections import defaultdict

import numpy as np

from utils import Params


class ReviewsData:
    def __init__(self, data_path, is_training=True):
        self.data_path = Path(data_path)
        self.X = []
        self.y = [] if is_training else None
        self.paths = []
        self.classes = Params().classes
        self.vocablist = Params().vocablist
        self.stopwords = Params().stopwords

    def read_txt(self):
        paths = self.data_path.glob('**/*.txt')
        for p in paths:
            if p.parts[-1] == 'README.txt':
                continue
            with open(p, 'r') as f:
                self.X.append(f.readlines())
            self.paths.append(p)
            if self.y is not None:
                truthful = 'truthful' if 'truthful' in p.parts[-3] else 'deceptive'
                positive = 'positive' if 'positive' in p.parts[-4] else 'negative'
                self.y.append(truthful + '_' + positive)

    def preprocess(self, n_largest=1000, use_vocab=True):
        self._normalize()
        self._tokenize(use_vocab)
        #self._feature_selection(n_largest)
    
    def _tokenize(self, use_vocab):
        for i, doc in enumerate(self.X):
            if use_vocab:
                doc = ' '.join([word for word in doc.split() if word in self.vocablist])
            else:
                doc = ' '.join([word for word in doc.split() if word not in self.stopwords])
            doc = doc.split(' ')
            self.X[i] = doc


    def _normalize(self):
        for i, doc in enumerate(self.X):
            doc = doc[0].lower()
            doc = re.sub(r'\d+','', doc)
            doc = re.sub(r'[^\w\s]', '', doc)
            doc = re.sub(r'[!@#$.,]', '', doc)
            doc = doc.strip()
            self.X[i] = doc

    def _feature_selection(self, n_largest):
        word_existence_per_doc = {cls:defaultdict(lambda:0) for cls in self.classes}

        for cls in self.classes:
            for doc in self.X:
                cur_words = set(doc)
                for word in cur_words:
                    word_existence_per_doc[cls][word] += 1

        top_words = {cls:None for cls in self.classes}
        for cls in self.classes:
            top = heapq.nlargest(n_largest, 
                                 word_existence_per_doc[cls], 
                                 word_existence_per_doc[cls].get)
            top_words[cls] = top
        for cls in self.classes:
            for i, (doc, label) in enumerate(zip(self.X, self.y)):
                if label == cls:
                    new_doc = []
                    for word in doc:
                        if word in top_words[cls]:
                            new_doc.append(word)
                    self.X[i] = new_doc
        

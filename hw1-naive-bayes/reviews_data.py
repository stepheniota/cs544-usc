import re
import heapq
from pathlib import Path
from collections import defaultdict

from utils import Params


class ReviewsData:
    def __init__(self, data_paths, is_training=True, fold_to_ignore=None, is_val=False):
        self.data_paths = data_paths
        self.X = []
        self.is_training = is_training
        self.y = [] if is_training else None
        self.paths = []
        self.classes = Params().classes
        self.vocablist = Params().vocablist
        self.stopwords = Params().stopwords
        self.switchwords = Params().switchwords

    def read_txt(self):
        for p in self.data_paths:
            with open(p, 'r') as f:
                self.X.append(f.readlines())
            self.paths.append(p)
            if self.y is not None:
                truthful = 'truthful' if 'truthful' in Path(p).parts[-3] else 'deceptive'
                positive = 'positive' if 'positive' in Path(p).parts[-4] else 'negative'
                self.y.append(truthful + '_' + positive)

    def preprocess(self, use_vocab=True, use_stop=True):
        self._normalize()
        self._tokenize(use_vocab, use_stop)
        #self._feature_selection(n_largest)
    
    def _tokenize(self, use_vocab, use_stop):
        for i, doc in enumerate(self.X):
            if use_vocab:
                doc = ' '.join([word for word in doc.split() if word in self.vocablist])
            if use_stop:
                doc = ' '.join([word for word in doc.split() if word not in self.stopwords])

            doc = ' '.join([self.switchwords[w] if w in self.switchwords else w for w in doc.split()])
            doc = doc.split()

            self.X[i] = doc


    def _normalize(self):
        for i, doc in enumerate(self.X):
            doc = doc[0].lower()
            doc = re.sub(r'[0-9]', ' ', doc)
            #doc = re.sub(r'\d+','', doc)
            #doc = re.sub(r'[^\w\s]', '', doc)
            doc = re.sub(r'[!@#$.,]', ' ', doc)
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
        

""" Naive Bayes classification model. """
import json
from collections import Counter, defaultdict

import numpy as np
from utils import Params


class NaiveBayesClassifer:
    """ Naive Bayes Classifier. """

    def __init__(self, n_classes=4, type='multinomial'):
        if type not in set(['multinomial', 'bernoulli']):
            raise NotImplementedError
        self.type = type
        if n_classes != 2 and n_classes != 4:
            raise NotImplementedError
        self.n_classes = n_classes
        if n_classes == 4:
            self.classes = Params().classes
        else:
            self.classes = ['truthful', 'deceptive', 'positive', 'negative']
        self.priors = {cls:0 for cls in self.classes}
        self.condprob = [defaultdict(lambda:0) for _ in self.classes]
        self.vocab = set()


    def fit(self, X, y):
        """ Fit NB classifier according to X, y. """
        for doc in X:
            self.vocab.update(doc)

        if self.type == 'multinomial':
            if self.n_classes == 4:
                self._fit_multi(X, y)
            else:
                self._fit_multi_2(X, y)
        else:
            if self.n_classes == 4:
                self._fit_bern(X, y)
            else:
                self._fit_bern_2(X, y)
            
        # normalize priors 
        for cls in self.classes:
            self.priors[cls] /= len(X)
    

    def _fit_bern(self, X, y):
        for i, cls in enumerate(self.classes):
            for word in self.vocab:
                for doc, y_label in zip(X, y):
                    if y_label == cls:
                        self.priors[cls] += 1
                        if word in doc:
                            self.condprob[i][word] += 1
        
        for word in self.vocab:
            for i, cls in enumerate(self.classes):
                if word not in self.condprob[i]:
                    self.condprob[i][word] += 1
        
        # smoothing
        for i, cls in enumerate(self.classes):
            self.condprob[i] = {word:(freq)/(self.priors[cls] + 2) 
                                for word, freq in self.condprob[i].items()}

    def _fit_bern_2(self, X, y):
        for i, cls in enumerate(self.classes):
            for word in self.vocab:
                for doc, y_label in zip(X, y):
                    if cls in y_label: 
                        self.priors[cls] += 1
                        if word in doc:
                            self.condprob[i][word] += 1

        for word in self.vocab:
            for i, cls in enumerate(self.classes):
                if word not in self.condprob[i]:
                    self.condprob[i][word] += 1

        # smoothing
        for i, cls in enumerate(self.classes):
            self.condprob[i] = {word:freq/(self.priors[cls] + 2) 
                                for word, freq in self.condprob[i].items()}


    def _fit_multi(self, X, y):
        for i, cls in enumerate(self.classes):
            for doc, y_label in zip(X, y):
                if y_label == cls:
                    self.priors[cls] += 1
                    cur_wordlist = Counter(doc)
                    for word, count in cur_wordlist.items():
                        self.condprob[i][word] += count

        # Laplace smoothing for all words
        for word in self.vocab:
            for i, cls_condprob in enumerate(self.condprob):
                cls_condprob[word] = (1 + cls_condprob[word])
                self.condprob[i] = cls_condprob
        
        for i, cls_condprob in enumerate(self.condprob):
            N = sum(cls_condprob.values())
            self.condprob[i] = {w:val/N for w, val in cls_condprob.items()}
            

    def _fit_multi_2(self, X, y):
        for i, cls in enumerate(self.classes):
            for doc, y_label in zip(X, y):
                if cls in y_label:
                    self.priors[cls] += 1
                    cur_wordlist = Counter(doc)
                    for word, count in cur_wordlist.items():
                        self.condprob[i][word] += count

        # Laplace smoothing for all words
        for word in self.vocab:
            for cls_condprob in self.condprob:
                cls_condprob[word] += 1

    def predict(self, X):
        """ Perform classification on feature matrix X. """

        y_hat = [None for _ in range(len(X))]

        if self.type == 'multinomial':
            if self.n_classes == 4:
                return self._predict_multi(y_hat, X)
            return self._predict_multi_2(y_hat, X)
        else:
            return self._predict_bern(y_hat, X)
        

    def _predict_multi(self, y_hat, X):
        for i_X, doc in enumerate(X):
            scores = np.zeros(self.n_classes, dtype=np.float128)
            for i_cls, cls in enumerate(self.classes):
                scores[i_cls] = np.log(self.priors[cls])
                for word in doc:
                    # exclude terms not in vocab
                    if word in self.vocab:
                        scores[i_cls] += np.log(self.condprob[i_cls][word])

            i_max = np.argmax(scores)
            y_hat[i_X] = self.classes[i_max]

        return y_hat

    def _predict_multi_2(self, y_hat, X):
        for i_X, doc in enumerate(X):
            scores = np.zeros(4, dtype=np.float128)
            for i_cls, cls in enumerate(self.classes):
                scores[i_cls] = np.log(self.priors[cls])
                for word in doc:
                    if word in self.vocab:
                        scores[i_cls] += np.log(self.condprob[i_cls][word])

            i_max_a = 0 if scores[0] > scores[1] else 1
            i_max_b = 2 if scores[2] > scores[3] else 3
            y_hat[i_X] = self.classes[i_max_a] + '_' + self.classes[i_max_b]

        return y_hat

    
    def _predict_bern(self, y_hat, X):
        for i_X, doc in enumerate(X):
            X_vocab = set(doc)
        
            scores = np.zeros(4, dtype=np.float128)
            for i_cls, cls in enumerate(self.classes):
                scores[i_cls] = np.log(self.priors[cls])
                for word in self.vocab: 
                    if word in X_vocab:
                        scores[i_cls] += np.log(self.condprob[i_cls][word])
                    else:
                        scores[i_cls] -= np.log(1 - self.condprob[i_cls][word])

            if self.n_classes == 4:
                i_max = np.argmax(scores)
                y_hat[i_X] = self.classes[i_max]
            else:
                i_max_a = 0 if scores[0] > scores[1] else 1
                i_max_b = 2 if scores[2] > scores[3] else 3
                y_hat[i_X] = self.classes[i_max_a] + '_' + self.classes[i_max_b]

        return y_hat


    def save_json(self, file_name='nbmodel.txt'):
        """ Write model params to human-readable txt file. """
        params = {}
        params['priors'] = self.priors
        params['type'] = self.type
        params['n_classes'] = self.n_classes
        params['classes'] = self.classes
        params['condprob'] = self.condprob
        params['vocab'] = list(self.vocab)
        
        json_txt = json.dumps(params, indent=4)
        with open(file_name, 'w') as f:
            f.write(json_txt)


    def load_json(self, file_name='nbmodel.txt'):
        """ Load model params previously saved using self.dump(). """ 
        with open(file_name, 'r') as f:
            params = json.load(f)

        self.priors = params['priors']
        self.type = params['type']
        self.n_classes = params['n_classes']
        self.classes = params['classes']
        self.condprob = params['condprob']
        self.vocab = set(params['vocab'])


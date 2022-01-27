""" Naive Bayes classification model. """
import json
from collections import Counter, defaultdict

import numpy as np
from utils import Params


class NaiveBayesClassifer:
    """ Naive Bayes Classifier. """

    def __init__(self, n_classes=4, type='multinomial'):
        if type != 'multinomial':
            raise NotImplementedError
        self.type = type
        if n_classes != 2 and n_classes != 4:
            raise NotImplementedError
        self.n_classes = n_classes
        self.classes = Params().classes
        self.priors = {cls:0 for cls in self.classes}
        self.condprob = [defaultdict(lambda:0) for _ in self.classes]
        self.vocab = set()


    def fit(self, X, y):
        """ Fit NB classifier according to X, y. """
        # extract vocab
        for doc in X:
            self.vocab.update(doc)

        for i, cls in enumerate(self.classes):
            for doc, y_label in zip(X, y):
                if y_label == cls:
                    self.priors[cls] += 1
                    cur_wordlist = Counter(doc)
                    for word, count in cur_wordlist.items():
                        self.condprob[i][word] += count

        # Laplace smoothing for all words
        for word in self.vocab:
            for cls_condprob in self.condprob:
                cls_condprob[word] += 1

        # normalize priors 
        for cls in self.classes:
            self.priors[cls] /= len(X)


    def predict(self, X):
        """ Perform classification on feature matrix X. """
        y_hat = [None for _ in range(len(X))]
        for i_X, doc in enumerate(X):
            scores = np.zeros(self.n_classes, dtype=np.float64)
            for i_cls, cls in enumerate(self.classes):
                scores[i_cls] = self.priors[cls]
                for term in doc:
                    # exclude terms not in vocab
                    if term in self.vocab:
                        scores[i_cls] += np.log(self.condprob[i_cls][term])

            i_max = np.argmax(scores)
            y_hat[i_X] = self.classes[i_max]

        return y_hat


    def save_json(self, file_name='nbmodel.txt'):
        """ Write model params to human-readable txt file. """
        params = {}
        params['priors'] = self.priors
        params['type'] = self.type
        params['n_classes'] = self.n_classes
        params['classes'] = self.classes
        #params['class_idx'] = self.class_idx
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
        #self.class_idx = params['class_idx']
        self.condprob = params['condprob']
        self.vocab = set(params['vocab'])


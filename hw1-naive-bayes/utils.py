""" Misc. functions that make my life easier. """
import numpy as np


class Params:
    def __init__(self):
        self.classes = ['truthful_positive', 'truthful_negative', 
                       'deceptive_positive', 'deceptive_negative']
        self.class_labels = {
            'truthful_positive':np.array((1, 0, 0, 0)),
            'truthful_negative':np.array((0, 1, 0, 0)),
            'deceptive_positive':np.array((0, 0, 1, 0)),
            'deceptive_negative':np.array((0, 0, 0, 1))
        }

def precision(true_pos, false_pos):
    return true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0

def recall(true_pos, false_neg):
    return true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0

def accuracy_score(y_true, y_pred):
    score = sum([true == pred for true, pred in zip(y_true, y_pred)])
    return score / len(y_true)


def f1_score(y_true, y_pred, n_classes=4):
    """ Calculates the mean f1 score across the four classes. """
    from nbmodel import NaiveBayesClassifer
    scores = np.zeros(4, np.float64)
    classes = NaiveBayesClassifer(n_classes).classes
    for i, cls in enumerate(classes):
        true_pos = false_neg = false_pos = 0
        for label, pred in zip(y_true, y_pred):
            if np.all(label == pred) and np.all(label == cls):
                true_pos += 1
            elif np.any(label != pred) and np.all(label == cls):
                false_neg += 1
            elif np.any(label != pred) and np.any(label != cls):
                false_pos + 1
        p = precision(true_pos, false_pos)
        r = recall(true_pos, false_neg)
        scores[i] = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.mean(scores)

def save_predictions(data_cls, y_preds, file_name='nboutput.txt'):
    with open(file_name, 'w') as f:
        for pred, path in zip(y_preds, data_cls.paths):
            labels = pred.split('_')
            f.write(f'{labels[0]} \t {labels[1]} \t {path} \n')


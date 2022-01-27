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
        self.stopwords = set(
            ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'dont', 'should', 'now']
        )

def precision(true_pos, false_pos):
    return true_pos / (true_pos + false_pos)

def recall(true_pos, false_neg):
    return true_pos / (true_pos + false_neg)

def accuracy_score(y_true, y_pred):
    score = sum([true == pred for true, pred in zip(y_true, y_pred)])
    return score / len(y_true)


def f1_score(y_true, y_pred, n_classes=4):
    """ Calculates the mean f1 score across the four classes. """
    #from nbmodel import NaiveBayesClassifer
    scores = np.zeros(4, np.float64)
    classes = ['truthful', 'deceptive', 'positive', 'negative']
    for i, cls in enumerate(classes):
        true_pos, false_pos, false_neg = 0, 0, 0
        for true_labels, pred_labels in zip(y_true, y_pred):
            idx = 0 if i == 0 or i == 1 else 1
            true = true_labels.split('_')[idx]
            pred = pred_labels.split('_')[idx]
            if true == pred and true == cls:
                true_pos += 1
            elif true != pred and pred == cls:
                false_pos += 1
            elif true != pred and true == cls:
                false_neg += 1
        p = precision(true_pos, false_pos)
        r = recall(true_pos, false_neg)
        scores[i] = (2 * p * r) / (p + r)
    return np.mean(scores)

def save_predictions(data_cls, y_preds, file_name='nboutput.txt'):
    with open(file_name, 'w') as f:
        for pred, path in zip(y_preds, data_cls.paths):
            labels = pred.split('_')
            f.write(f'{labels[0]} \t {labels[1]} \t {path} \n')


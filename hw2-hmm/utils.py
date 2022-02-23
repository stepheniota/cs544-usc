"""Helper functions."""
import math
import random

import numpy as np

def normalize_dict(d, scale=None, in_place=True):
    """Normalize input dict s.t. it sums to one."""
    if not scale:
        scale = sum(d.values())
    if in_place:
        for key in d:
            d[key] /= scale
    else:
        return {key: val/scale for key, val in d.items()}

def logaddexp2(*args):
    out = float(0)
    for n in args:
        out += 2**n
    return np.log2(out)

def logaddexp(*args):
    out = float(0)
    for n in args:
        out += np.exp(n)
    return np.log(out)

def log_sum(*args):
    """Compute sum_i { log(args[i]) }"""
    out = float(0.0)
    for n in args:
        if n > 0:
            out += math.log(n)
    return out

def prod(*args):
    out = float(1.0)
    for n in args:
        if n > 0:
            out *= n
    return out

def accuracy_score(y, y_hat):
    score = sum(true == pred for i, j in zip(y, y_hat) 
                for true, pred in zip(i, j))
    return score / sum(len(yy) for yy in y)

def compare_output(manual_path, tagged_path):
    total = correct = 0
    with open(manual_path, mode='r') as f:
        manual_data = f.readlines()
    with open(tagged_path, mode='r') as f:
        tagged_data = f.readlines()
    for x, y in zip(manual_data, tagged_data):
        total += 1
        if x == y: correct += 1
    return correct / total

def write_output(X, y_hat, file_name="hmmoutput.txt"):
    with open(file_name, mode='w') as f:
        for word_sentence, tag_sentence in zip(X, y_hat):
            for i, (word, tag) in enumerate(zip(word_sentence, tag_sentence)):
                f.write(word + '/' + tag)
                if len(word_sentence) - 1 == i:
                    f.write('\n')
                else:
                    f.write(' ')

def shuffle_data(X, y):
    tmp = list(zip(X, y))
    random.shuffle(tmp)
    return zip(*tmp)

if __name__ == "__main__":
    print(log_sum(1, 2, 3, 4, 5))
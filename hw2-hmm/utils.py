"""Helper functions."""
import numpy as np

def normalize_dict(d, scale=None, in_place=True):
    """Normalize input dict s.t. it sums to one."""
    if not scale:
        scale = sum(d.values())
    if 0 == scale:
        return
    if in_place:
        for key in d:
            d[key] /= scale
        return
    else:
        return {key: val/scale for key, val in d.items()}

def logaddexp2(*nums):
    return np.log2(sum(2**n for n in nums))

def logaddexp(*nums):
    return np.log(sum(np.exp(n) for n in nums))

def logsum(*nums):
    return sum(np.log(n) for n in nums)

def prod(*nums):
    return np.product(nums)

def accuracy_score(y, y_hat):
    score = sum(true == pred for i, j in zip(y, y_hat)
                for true, pred in zip(i, j))
    return score / sum(len(yy) for yy in y)

def write_output(X, y_hat, file_name="hmmoutput.txt"):
    with open(file_name, mode='w') as f:
        for word_sentence, tag_sentence in zip(X, y_hat):
            for i, (word, tag) in enumerate(zip(word_sentence, tag_sentence)):
                f.write(f"{word}/{tag}")  # f.write(word + '/' + tag)
                if len(word_sentence) - 1 == i:
                    f.write('\n')
                else:
                    f.write(' ')

def shuffle_data(X, y):
    import random
    tmp = list(zip(X, y))
    random.shuffle(tmp)
    return zip(*tmp)

if __name__ == "__main__":
    a = logsum(2, 3, 4, 5)
    print(type(a))

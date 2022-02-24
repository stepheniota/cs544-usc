"""Learn HMM parameters with fully-observed, tokenized training data."""
import sys
from pathlib import Path

import utils
from utils import prod, logaddexp, logaddexp2, logsum
from hmmmodel import HMM
from pos_data import POSData


def train_hmm(input_path, n_open=5):
    data = POSData(input_path, train=True)
    hmm = HMM()
    hmm.fit(data.X, data.y, n_open=n_open)
    hmm.save_params()


def test_hmm(input_path):
    data = POSData(input_path, train=False)
    hmm = HMM()
    hmm.load_params()
    y_hat = hmm.decode(data.X, prod)
    return y_hat


def dev_hmm():
    data_path = Path('hmm-training-data')
    paths = [['it_isdt_train_tagged.txt', 'it_isdt_dev_raw.txt', 'it_isdt_dev_tagged.txt'],
             ['ja_gsd_train_tagged.txt', 'ja_gsd_dev_raw.txt', 'ja_gsd_dev_tagged.txt']]
    train_idx, dev_idx, ref_idx = 0, 1, 2

    for lang in paths:
        ref_data = POSData(data_path/lang[ref_idx], train=True)
        best_score = 0.
        for n in range(1, 10):
            train_hmm(data_path/lang[train_idx], n)
            y_hat = test_hmm(data_path/lang[dev_idx])

            score = utils.accuracy_score(y=ref_data.y, y_hat=y_hat)
            print(f'\t Language: {lang[0][:2]} \t n_open: {n} \t Accuracy: {score:0.5f}')

        print('')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_hmm(sys.argv[1])
    else:
        dev_hmm()

    #train_hmm(sys.argv[1]) if len(sys.argv) > 1 else dev_hmm()

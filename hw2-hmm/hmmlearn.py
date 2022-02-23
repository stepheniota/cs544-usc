"""Learn HMM parameters with fully-observed, tokenized training data."""
import sys
from pathlib import Path

import utils
from hmmmodel import HMM
from pos_data import POSData


def train_hmm(input_path):
    data = POSData(input_path, train=True)
    hmm = HMM()
    hmm.fit(data.X, data.y)
    hmm.save_params()


def test_hmm(input_path):
    data = POSData(input_path, train=False)
    hmm = HMM()
    hmm.load_params()
    y_hat = hmm.decode(data.X)
    return hmm.decode(data.X)


def dev_hmm():
    data_path = Path('hmm-training-data')
    paths = [['it_isdt_train_tagged.txt', 'it_isdt_dev_raw.txt', 'it_isdt_dev_tagged.txt'],
             ['ja_gsd_train_tagged.txt', 'ja_gsd_dev_raw.txt', 'ja_gsd_dev_tagged.txt']]
    train_idx, dev_idx, ref_idx = 0, 1, 2

    for lang in paths:
        train_hmm(data_path/lang[train_idx])
        y_hat = test_hmm(data_path/lang[dev_idx])

        ref_data = POSData(data_path/lang[ref_idx], train=True)
        score = utils.accuracy_score(y=ref_data.y, y_hat=y_hat)
        print(f'\t Language: {lang[0][:2]} \t Accuracy: {score:0.10f}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_hmm(sys.argv[1])
    else:
        dev_hmm()

    #train_hmm(sys.argv[1]) if len(sys.argv) > 1 else dev_hmm()

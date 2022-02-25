"""Learn HMM parameters with fully-observed, tokenized training data."""
import sys
from pathlib import Path

from utils import accuracy_score, write_output
from hmmmodel import HMM
from pos_data import POSData


def train_hmm(input_path, n_open=4, bias=6, smooth=1):
    data = POSData(input_path, train=True)
    hmm = HMM()
    hmm.fit(data.X, data.y, n_open=n_open, open_bias=bias, smooth=smooth)
    hmm.save_params()


def test_hmm(input_path):
    data = POSData(input_path, train=False)
    hmm = HMM()
    hmm.load_params()
    y_hat = hmm.decode(data.X)
    write_output(data.X, y_hat)

    return y_hat


def dev_hmm():
    data_path = Path('hmm-training-data')
    paths = [['it_isdt_train_tagged.txt', 'it_isdt_dev_raw.txt', 'it_isdt_dev_tagged.txt'],
             ['ja_gsd_train_tagged.txt', 'ja_gsd_dev_raw.txt', 'ja_gsd_dev_tagged.txt']]
    train_idx, dev_idx, ref_idx = 0, 1, 2

    for lang in paths:
        ref_data = POSData(data_path/lang[ref_idx], train=True)
        scores = []
        for n in range(1, 10):
            for bias in range(10):
                for smooth in range(2, 20, 2):
                    train_hmm(data_path/lang[train_idx], n, bias, smooth)
                    y_hat = test_hmm(data_path/lang[dev_idx])

                    score = accuracy_score(y=ref_data.y, y_hat=y_hat)
                    #print(f'\t Language: {lang[0][:2]} \t n_open: {n} \t bias: {bias} \t Accuracy: {score:0.4f}')
                    scores.append([score, n, bias, smooth])

        TOP = 20
        scores.sort(key=lambda x: x[0], reverse=True)
        print(f"Language {lang[0][:2]} top {TOP}:")
        for (score, n, bias, smooth) in scores[:TOP]:
            print(f"\t score: {score:0.4f} \t n: {n} \t bias: {bias} \t smooth: {smooth}")
        print('')


if __name__ == '__main__':
    train_hmm(sys.argv[1]) if len(sys.argv) > 1 else dev_hmm()

import sys
from pathlib import Path

import hmmmodel
import utils
from pos_data import POSData


def train_hmm(input_path, is_dev=False):
    data = POSData(input_path, is_training=is_dev)
    data.read_txt()

    hmm = hmmmodel.HMM(is_training=is_dev)
    hmm.fit(data.X, data.y)
    hmm.save_json()


def test_hmm(input_path, ):
    data = POSData(input_path, is_training=False)
    data.read_txt()

    hmm = hmmmodel.HMM(is_training=False)
    hmm.load_json()
    return hmm.decode(data.X)


def dev_hmm():
    data_path = Path('hmm-training-data')
    paths = [
        ['it_isdt_train_tagged.txt', 'it_isdt_dev_raw.txt', 'it_isdt_dev_tagged.txt'],
        ['ja_gsd_train_tagged.txt', 'ja_gsd_train_tagged.txt', 'ja_gsd_train_tagged.txt']
    ]
    train_idx, dev_idx, ref_idx = 0, 1, 2

    for lang in paths:
        la = lang[0][:2]
        train_hmm(data_path/lang[train_idx], is_dev=True)

        y_hat = test_hmm(data_path/lang[dev_idx])

        ref_data = POSData(data=data_path/lang[ref_idx], is_training=True)
        ref_data.read_txt()

        score = utils.accuracy_score(ref_data.y, y_hat)
        print(f'\t Language: {la} \t Accuracy: {score}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_hmm(sys.argv[1])
    else:
        dev_hmm()
    #train_hmm(sys.argv[1]) if len(sys.argv) > 1 else dev_hmm()


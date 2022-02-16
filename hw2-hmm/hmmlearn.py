"""Training script to learn HMM parameters."""

import sys
from pathlib import Path
from typing import Union, List

import utils
from hmmmodel import HMM
from pos_data import POSData


def train_hmm(input_path: Union[Path, str], is_dev: bool = False) -> None:
    data = POSData(input_path, train=is_dev, read=True)

    hmm = HMM(train=is_dev)
    hmm.fit(data.X, data.y)
    hmm.save_json()


def test_hmm(input_path: Path) -> List[List[str]]:
    data = POSData(input_path, train=False, read=True)
    #data.read_txt()

    hmm = HMM(train=False)
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

        ref_data = POSData(data_path/lang[ref_idx], train=True)
        #ref_data.read_txt()

        score = utils.accuracy_score(ref_data.y, y_hat)
        print(f'\t Language: {la} \t Accuracy: {score}')

    return

def get_baselines():
    from baseline_tagger import BaselineTagger

    data_path = Path('hmm-training-data')
    paths = [
        ['it_isdt_train_tagged.txt', 'it_isdt_dev_raw.txt', 'it_isdt_dev_tagged.txt'],
        ['ja_gsd_train_tagged.txt', 'ja_gsd_train_tagged.txt', 'ja_gsd_train_tagged.txt']
    ]
    train_idx, dev_idx, ref_idx = 0, 1, 2

    for lang in paths:
        la = lang[0][:2]
        train_data = POSData(data_path/lang[train_idx], train=True)
        dev_data = POSData(data_path/lang[dev_idx], train=False)
        ref_data = POSData(data_path/lang[ref_idx], train=True)

        continue
        tagger = BaselineTagger()
        tagger.fit(train_data.X, train_data.y)

        y_hat = tagger.decode(dev_data.X)

        score = utils.accuracy_score(ref_data.y, y_hat)
        print(f'Baseline Accuracy. Language: {la} \t Accuracy: {score}')

    return


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_hmm(sys.argv[1])
    else:
        #get_baselines()
        dev_hmm()

    #train_hmm(sys.argv[1]) if len(sys.argv) > 1 else dev_hmm()


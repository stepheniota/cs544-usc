"""Test script to decode tokenized text using learned HMM params."""

import sys

from hmmmodel import HMM
from pos_data import POSData


def decode_hmm(data_path):
    data = POSData(data_path, is_training=False)
    data.read_txt()

    hmm = HMM(is_training=False)
    hmm.load_json()
    y_hat = hmm.decode(data.X)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        decode_hmm(sys.argv[1])
    else:
        raise NotImplementedError
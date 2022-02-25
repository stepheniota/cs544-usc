"""Decode tokenized text using learned HMM params."""
import sys

from hmmmodel import HMM
from pos_data import POSData
from utils import write_output


def decode_hmm(data_path):
    data = POSData(data_path, train=False)

    hmm = HMM()
    hmm.load_params()
    y_hat = hmm.decode(data.X)

    write_output(data.X, y_hat)


if __name__ == '__main__':
    assert len(sys.argv) > 1 and "Must provide argument `/path/to/input`."
    decode_hmm(sys.argv[1])

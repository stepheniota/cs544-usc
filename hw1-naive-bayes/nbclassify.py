import os
import sys
from glob import glob
from pathlib import Path

from nbmodel import NaiveBayesClassifer
from utils import save_predictions
from reviews_data import ReviewsData


def main(data_path):
    data_paths = glob(os.path.join(str(data_path), '*/*/*/*.txt'))

    test_data = ReviewsData(data_paths, is_training=False)
    test_data.read_txt()
    test_data.preprocess(use_vocab=False, use_stop=True)

    model = NaiveBayesClassifer()
    model.load_json()

    y_pred = model.predict(test_data.X)

    save_predictions(test_data, y_pred)


if __name__ == '__main__':
    data_path = 'op_spam_training_data' if len(sys.argv) == 1 else sys.argv[1]
    main(data_path)

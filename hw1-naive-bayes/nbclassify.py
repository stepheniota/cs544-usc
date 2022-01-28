import sys
from pathlib import Path

from nbmodel import NaiveBayesClassifer
from utils import save_predictions, f1_score
from reviews_data import ReviewsData


def main(data_path):
    test_data = ReviewsData(data_path)
    test_data.read_txt()
    test_data.preprocess()

    model = NaiveBayesClassifer()
    model.load_json()

    y_pred = model.predict(test_data.X)

    save_predictions(test_data, y_pred)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        data_path = Path('op_spam_training_data')
    else:
        data_path = Path(sys.argv[1])
    main(data_path)

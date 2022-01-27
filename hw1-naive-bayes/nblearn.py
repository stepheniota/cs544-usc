import sys
from pathlib import Path

import numpy as np

from utils import f1_score, save_predictions, accuracy_score
from nbmodel import NaiveBayesClassifer
from reviews_data import ReviewsData


def train(data_path, is_testing):
    train_data = ReviewsData(data_path, is_training=True)
    train_data.read_txt()
    train_data.preprocess()

    nb_model = NaiveBayesClassifer(type='bernoulli')
    nb_model.fit(train_data.X, train_data.y)

    nb_model.save_json()

    if is_testing:
        y_pred_train = nb_model.predict(train_data.X)
        score = f1_score(train_data.y, y_pred_train)
        print(f'mean f1 score on train set = {score}')
        #print(f'accuracy on train set = {accuracy_score(train_data.y, y_pred_train)}')


        val_data = ReviewsData(Path('op_spam_test_data'), is_training=True)
        val_data.read_txt()
        val_data.preprocess()

        y_pred = nb_model.predict(val_data.X)
        score = f1_score(val_data.y, y_pred)
        print(f'mean f1 score on val set = {score}')
        #print(f'accuracy on train set = {accuracy_score(val_data.y, y_pred)}')
        save_predictions(val_data, y_pred)


if __name__ == '__main__':
    is_testing = True if len(sys.argv) < 2 else False

    data_path = Path(sys.argv[1]) if not is_testing else Path('op_spam_training_data')

    train(data_path, is_testing)
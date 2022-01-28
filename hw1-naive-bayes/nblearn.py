import sys
from pathlib import Path

import numpy as np

from utils import f1_score, save_predictions, accuracy_score
from nbmodel import NaiveBayesClassifer
from reviews_data import ReviewsData


def train(data_path, is_testing):
    train_data = ReviewsData(data_path, is_training=True)
    train_data.read_txt()
    train_data.preprocess(use_vocab=True)

    nb_model = NaiveBayesClassifer(type='multinomial', n_classes=4)
    nb_model.fit(train_data.X, train_data.y)

    best_score = float('-inf')
    best_model = None
    best_n = float('-inf')
    x = """
    for n in range(50, 1000, 10):
        model = NaiveBayesClassifer(type='bernoulli')

        train_data = ReviewsData(data_path, is_training=True)
        train_data.read_txt()
        train_data.preprocess(n_largest=n)
        val_data = ReviewsData(Path('op_spam_test_data'), is_training=True)
        val_data.read_txt()
        val_data.preprocess(n_largest=n)

        nb_model = NaiveBayesClassifer(type='bernoulli')
        nb_model.fit(train_data.X, train_data.y)
        y_pred = nb_model.predict(val_data.X)
        cur_score = f1_score(val_data.y, y_pred)
        print(f'n: {n} \t score: {cur_score}')
        if cur_score >= best_score:
            best_score, best_model = cur_score, model
            best_n = n

    print(f'best n = {best_n}, best score = {best_score}')
    """


    nb_model.save_json()
    #if best_model is not None:
    #    best_model.save_json()

    if is_testing:
        y_pred_train = nb_model.predict(train_data.X)
        score = f1_score(train_data.y, y_pred_train)
        print(f'mean f1 score on train set = {score}')
        print(f'accuracy on train set = {accuracy_score(train_data.y, y_pred_train)}')


        val_data = ReviewsData(Path('op_spam_test_data'), is_training=True)
        val_data.read_txt()
        val_data.preprocess(use_vocab=True)

        y_pred = nb_model.predict(val_data.X)
        score = f1_score(val_data.y, y_pred)
        print(f'mean f1 score on val set = {score}')
        #print(f'accuracy on train set = {accuracy_score(val_data.y, y_pred)}')
        save_predictions(val_data, y_pred)


if __name__ == '__main__':
    is_testing = True if len(sys.argv) < 2 else False

    data_path = Path(sys.argv[1]) if not is_testing else Path('op_spam_training_data')

    train(data_path, is_testing)
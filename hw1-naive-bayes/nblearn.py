import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

from utils import f1_score, save_predictions, accuracy_score
from nbmodel import NaiveBayesClassifer
from reviews_data import ReviewsData


def train(data_path, is_testing):
    data_paths = glob(os.path.join(str(data_path), '*/*/*/*.txt'))
    train_data = ReviewsData(data_paths, is_training=True)
    train_data.read_txt()
    train_data.preprocess(use_vocab=False, use_stop=True)

    nb_model = NaiveBayesClassifer(type='multinomial', n_classes=4)
    nb_model.fit(train_data.X, train_data.y)
    nb_model.save_json()

    if is_testing:
        folds = ['fold1', 'fold2', 'fold3', 'fold4']
        scores = [0 for _ in folds]
        for i, fold in enumerate(folds):
            train_paths_all = glob(os.path.join(str(data_path), '*/*/*/*.txt'))
            train_paths = []
            for p in train_paths_all:
                if fold != Path(p).parts[-2]:
                    train_paths.append(p)
            val_paths = glob(os.path.join(str(data_path), f'*/*/{fold}/*.txt'))

            train_data = ReviewsData(train_paths, is_training=True)
            train_data.read_txt()
            train_data.preprocess(use_vocab=False, use_stop=True)
            val_data = ReviewsData(val_paths, is_training=True)
            val_data.read_txt()
            val_data.preprocess(use_vocab=False, use_stop=True)

            model = NaiveBayesClassifer(type='multinomial', n_classes=4)
            model.fit(train_data.X, train_data.y)
            y_pred = model.predict(val_data.X)

            scores[i] = f1_score(val_data.y, y_pred)
            print(f'\t{fold} = {scores[i]}')

        print(f'model type = {model.type}')
        print(f'n classes = {model.n_classes}')
        print(f'average score = {np.mean(scores)}')


if __name__ == '__main__':
    is_testing = True if len(sys.argv) < 2 else False

    data_path = sys.argv[1] if not is_testing else 'op_spam_training_data'

    train(data_path, is_testing)
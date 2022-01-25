from pathlib import Path

import numpy as np

class ReviewsData:
    def __init__(self, data_path, is_training=True):
        self.data_path = Path(data_path)
        self.is_training = is_training
        self.data = []
        self.labels = []

    def read_txt(self):
        paths = self.data_path.glob('**/*.txt')
        for p in paths:
            if p.parts[-1] == 'README.txt':
                continue
            try:
                self.data.append(np.loadtxt(p, dtype=str))
            except Exception as e:
                print(f'{p} failed due to exception {e}')
            positive = 'positive' in p.parts[-4]
            truthful = 'truthful' in p.parts[-3]
            self.labels.append(np.array([positive, truthful]))

    def preprocess(self):
        pass
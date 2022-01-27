from pathlib import Path

import numpy as np

class ReviewsData:
    def __init__(self, data_path, is_training=True):
        self.data_path = Path(data_path)
        self.X = []
        self.y = [] if is_training else None
        self.paths = []

    def read_txt(self):
        paths = self.data_path.glob('**/*.txt')
        for p in paths:
            if p.parts[-1] == 'README.txt':
                continue
            try:
                self.X.append(np.loadtxt(p, dtype=str))
            except Exception as e:
                print(f'{p} failed due to exception {e}')
            self.paths.append(p)
            if self.y is not None:
                truthful = 'truthful' if 'truthful' in p.parts[-3] else 'deceptive'
                positive = 'positive' if 'positive' in p.parts[-4] else 'negative'
                self.y.append(truthful + '_' + positive)

    def preprocess(self):
        pass
from pathlib import Path

import numpy as np

class ReviewsData:
    def __init__(self, data_path, is_training=True):
        self.data_path = Path(data_path)
        self.data = []
        self.labels = [] if is_training else None
        self.paths = [] if is_training else None

    def read_txt(self):
        paths = self.data_path.glob('**/*.txt')
        for p in paths:
            if p.parts[-1] == 'README.txt':
                continue
            try:
                self.data.append(np.loadtxt(p, dtype=str))
            except Exception as e:
                print(f'{p} failed due to exception {e}')
            if self.labels and self.paths:
                positive = 'positive' if 'positive' in p.parts[-4] else 'negative'
                truthful = 'truthful' if 'truthful' in p.parts[-3] else 'deceptive'
                self.labels.append([positive, truthful])
                self.paths.append(p)

    def preprocess(self):
        pass
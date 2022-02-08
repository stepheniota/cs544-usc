import re 
from pathlib import Path

import numpy as np

class POSData:
    def __init__(self, data, is_training):
        self.data = data
        self.is_training = is_training
        self.X = []
        self.y = []

    def read_txt(self,):
        if self.is_training:
            with open(self.data, mode='r') as f:
                for line in f:
                    cur_X, cur_y = [], []
                    elle = line.split()
                    for word in elle:
                        word_tag = word.split('/')
                        cur_X.append(word_tag[0])
                        cur_y.append(word_tag[1])
                    self.X.append(cur_X)
                    self.y.append(cur_y)
        else:
            with open(self.data, mode='r') as f:
                for line in f:
                    self.X.append(line.split())
            #    for line in f:
            #        elle = line.split()



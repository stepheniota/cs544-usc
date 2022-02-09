"""Model class."""
import json

import numpy as np

class HMM:
    """Hidden Markov model with multinomial (discrete) emissions."""
    def __init__(self, is_training=True):
        self.is_training = is_training
    
    def fit(self, X, y):
        """Estimate model parameters."""
        pass

    def predict(self, X):
        raise NotImplementedError

    def decode(self, X):
        """Viterbi algorithm. 
        Find the most likely state sequence corresponding to X. 
        """
        raise NotImplementedError

    def save_json(self, file_name='hmmmodel.txt'):
        """Write model params to human-interpretable, json-formatted txt file."""
        json_txt = json.dumps(vars(self), indent=4)
        with open(file_name, mode='w') as f:
            f.write(json_txt)

    def load_json(self, file_name='hmmmodel.txt'):
        """Load pretrained model params from a json-formatted file."""
        with open(file_name, mode='r') as f:
            params = json.load(f)

        for key, val in params.items():
            setattr(self, key, val)
            #self.key = val
        

if __name__ == '__main__':
    hmm = HMM()

    hmm.load_json()
    print(vars(hmm))



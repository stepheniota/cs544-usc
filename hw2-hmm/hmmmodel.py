"""Hidden Markov model class."""
import json
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np

class HMM:
    """Hidden Markov model with discrete (i.e. multinomial) emissions."""
    def __init__(self, train: bool = True) -> None:
        self.train: bool = train
        self.states: set = set()
        self.obs: set = set()
        self.initial_probs: dict = dict()  # assume categorical dist.
        self.transitions: dict = dict()    # assume multinomial dist.
        self.emissions: dict = dict()      # assume multinomial dist.
        self.normalize: int = 0

    def fit(self, X: List[List[str]], y: List[List[str]]) -> None:
        """Estimate HMM parameters."""
        self._init_params(X, y)
        self._fit_params(X, y)

        # TODO: Lazy normalization? i.e. normalize during decoding
        #       Or do I even need to normalize? Not sure.
        self.normalize = sum(
            Counter(state for state_seq in y for state in state_seq).values()
        )

    def _init_params(self, X, y):
        """Initialize state space, obs space and priors."""
		# create state space from training data
        self.states = set(state for state_seq in y for state in state_seq)

        # initial probabilities: plus-one smoothing 
        self.initial_probs = {state: 1 for state in self.states}
        # no smoothing for transitions or emissions?
        self.transitions = {state: defaultdict(lambda: 0) 
                            for state in self.states}
        self.emissions = {state: defaultdict(lambda: 0) 
                            for state in self.states}

        # Do I even need to save the obs space separately?
        # Already included in emissions.values().keys()
        if self.obs is None:
            self.obs = set()

    def _fit_params(self, X, y):
        for obs_seq, state_seq in zip(X, y):
            for i, (obs, state) in enumerate(zip(obs_seq, state_seq)):
                self.obs.update(obs)
                if 0 == i:
                    self.initial_probs[state] += 1
                else:
                    self.transitions[state_seq[i-1]][state] += 1
                self.emissions[state][obs] += 1


    def decode(self, X: List[List[str]]) -> List[List[str]]:
        """Find the most likely state sequence corresponding to X. 
        Uses the Viterbi decoding algorithm.
        """
        return [self._viterbi(obs_seq) for obs_seq in X]

    def _viterbi(self, obs: List[str]) -> List[str]:
        raise NotImplementedError

    def save_json(self, file_name: str = "hmmmodel.txt") -> None:
        """Write model params to human-interpretable, json-format txt file."""
        params = vars(self)
        for key, var in params.items():
            
            if isinstance(var, set):  
                # sets can't be converted to json, go figure
                print('TODO: convert sets to list, then back to set upon load.')
                params[key] = list(var)
            
        json_txt = json.dumps(vars(self), indent=4)
        with open(file_name, mode='w') as f:
            f.write(json_txt)

    def load_json(self, file_name: str = "hmmmodel.txt") -> None:
        """Load pretrained model params from a json-formatted file."""
        with open(file_name, mode='r') as f:
            params = json.load(f)

        for key, val in params.items():
            if "states" == key or "obs" == key:
                val = set(val)
            setattr(self, key, val)
        

if __name__ == '__main__':
    hmm = HMM()

    hmm.load_json()
    print(vars(hmm))



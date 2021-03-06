"""Hidden Markov model class."""
import json

import numpy as np

from utils import normalize_dict

class HMM:
    """Hidden Markov model with categorical (i.e. discrete) emissions."""
    def __init__(self, end_st=True):
        self.states = None
        self.open_states = None
        self.obs = None
        self.init_probs = None  # assume categorical dist.
        self.trans = None  # assume categorical dist.
        self.emiss = None  # assume categorical dist.
        self.end_st = "END" if end_st else None

    def fit(self, X, y, n_open=5, open_bias=4, smooth=1):
        """Estimate HMM parameters."""
        self._init_params(X, y, smooth)
        self._fit_params(X, y)
        self._open_states(n_open, open_bias)
        self._normalize_params()

    def _open_states(self, n_open, open_bias):
        """Determine open states for unseen obs."""
        if n_open is None or n_open >= len(self.states):
            self.open_states = {st: 1 for st in self.states}
        else:
            vocab = {st: len(obs) for st, obs in self.emiss.items()}
            vocab_list = sorted(list(vocab), key=vocab.get, reverse=True)
            vocab_list = vocab_list[:n_open]
            if open_bias:
                self.open_states = {st: open_bias if st in vocab_list else 1
                                    for st in self.states}
            else:
                self.open_states = {st: 1 for st in vocab_list}

    def _init_params(self, X, y, smooth):
        """Initialize state space, obs space and priors."""
        self.states = set(st for state_seq in y for st in state_seq)
        self.states = list(self.states)
        if self.end_st:
            self.states.append(self.end_st)
            #self.states.insert(self.start_st)
        self.obs = set(obs for obs_seq in X for obs in obs_seq)
        # NOTE: plus-one smoothing for all possible combinations.
        self.init_probs = {st: 1 for st in self.states}
        self.trans = {prev_st: {st: smooth for st in self.states} for prev_st in self.states}
        self.emiss = {st: {} for st in self.states}

    def _fit_params(self, X, y):
        """MAP estimation of HMM probability distributions."""
        for i, (obs_seq, state_seq) in enumerate(zip(X, y)):
            for i, (obs, st) in enumerate(zip(obs_seq, state_seq)):
                if 0 == i:
                    self.init_probs[st] += 1
                else:
                    prev_st = state_seq[i-1]
                    self.trans[prev_st][st] += 1
                if obs in self.emiss[st]:
                    self.emiss[st][obs] += 1
                else:
                    self.emiss[st][obs] = 1
            if self.end_st:
                self.trans[st][self.end_st] += 1

    def _normalize_params(self):
        """All probability distributions must sum to one."""
        normalize_dict(self.init_probs)
        normalize_dict(self.open_states)

        for st in self.states:
            normalize_dict(self.trans[st])
            normalize_dict(self.emiss[st])

    def decode(self, X):
        """Find the most likely (hidden) state seq given observation seq X,
        through the Viterbi decoding algorithm.
        """
        return [self.viterbi(obs_seq)  for obs_seq in X]

    def viterbi(self, obs_seq):
        """Viterbi decoding algorithm."""
        T = len(obs_seq)
        if self.end_st:
            T += 1
        v_prob = np.zeros((T, len(self.states)))
        v_ptrs = {t: {} for t in range(T)}

        # base case
        init_obs = obs_seq[0]
        for i, st in enumerate(self.states):
            v_ptrs[0][st] = None
            init_pr = self.init_probs[st]
            if init_obs in self.emiss[st]:
                emiss_pr = self.emiss[st][init_obs]
                v_prob[0][i] = self._LLH_fn(init_pr, emiss_pr)
            elif init_obs not in self.obs and st in self.open_states:
                open_pr = self.open_states[st]
                v_prob[0][i] = self._LLH_fn(init_pr, open_pr)
            else:
                v_prob[0][i] = 0

        # forward pass
        for t, obs in enumerate(obs_seq[1:], start=1):
            for i, st in enumerate(self.states):
                if obs in self.emiss[st]:
                    emiss = self.emiss[st][obs]
                    best_pr, best_st = self._likelihood(v_prob, st, i, t, emiss)
                elif obs not in self.obs and st in self.open_states:
                    emiss = self.open_states[st]
                    best_pr, best_st = self._likelihood(v_prob, st, i, t, emiss)
                else:
                    best_pr, best_st = 0, None

                v_prob[t][i] = best_pr
                v_ptrs[t][st] = best_st

        if self.end_st:
            i = len(self.states) - 1
            st = self.end_st
            best_pr, best_st = self._likelihood(v_prob, st, i, T-1)
            v_prob[T-1][i] = best_pr
            v_ptrs[T-1][st] = best_st

        last_st_idx = np.argmax(v_prob[T-1])
        last_st = self.states[last_st_idx]

        best_path = self._viterbi_bkw(v_ptrs, last_st, T)

        return best_path

    def _viterbi_bkw(self, v_ptrs, last_st, T):
        best_path = [] if self.end_st else [last_st]
        for t in range(T-1, 0, -1):
            try:
                last_st = v_ptrs[t][last_st]
            except KeyError:
                print(f"t={t}")
                exit()
            best_path.append(last_st)

        return best_path[::-1]

    def _likelihood(self, v_prob, st, i_st, t, emiss=None):
        best_st, best_pr = None, np.float("-inf")
        for i_prev_st, prev_st in enumerate(self.states):
            trans = self.trans[prev_st][st]
            probsofar = v_prob[t-1][i_prev_st]
            if 0 == trans or 0 == probsofar:
                cur_pr = float("-inf")
            elif emiss and 0 != emiss:
                cur_pr = self._LLH_fn(trans, probsofar, emiss)
            else:
                cur_pr = self._LLH_fn(trans, probsofar)
            if cur_pr > best_pr:
                best_pr, best_st = cur_pr, prev_st

        return best_pr, best_st

    def _LLH_fn(self, *args):
        return np.product(args)

    def save_params(self, file_name="hmmmodel.txt"):
        """Write model params to human-interpretable, json-format txt file."""
        params = vars(self)  # NOTE: dangerous, doesn't create copy
        for attr_name, val in params.items():
            if isinstance(val, set):
                # NOTE: sets can't be converted to json, go figure
                params[attr_name] = list(val)

        json_txt = json.dumps(vars(self), indent=4)
        with open(file_name, mode='w', encoding="utf-8") as f:
            f.write(json_txt)

    def load_params(self, file_name="hmmmodel.txt"):
        """Load model params from a json-formatted txt file."""
        with open(file_name, mode='r', encoding="utf-8") as f:
            params = json.load(f)

        for attr_name, val in params.items():
            if isinstance(val, list) and attr_name != "states":
                val = set(val)
            setattr(self, attr_name, val)

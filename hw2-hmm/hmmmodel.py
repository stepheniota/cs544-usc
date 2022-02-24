"""Hidden Markov model class."""
import json

from utils import *

class HMM:
    """Hidden Markov model with categorical (i.e. discrete) emissions."""
    def __init__(self):
        self.states = None
        self.open_states = None
        self.obs = None
        self.init_probs = None  # assume categorical dist.
        self.trans = None  # assume categorical dist.
        self.emiss = None  # assume categorical dist.

    def fit(self, X, y, n_open=4):
        """Estimate HMM parameters."""
        self._init_params(X, y)
        self._fit_params(X, y)
        self._open_states(n_open)
        self._normalize_params()

    def _open_states(self, n_open):
        """Determine open states for unseen obs."""
        vocab = {st: len(obs) for st, obs in self.emiss.items()}
        vocab_list = sorted(list(vocab), key=vocab.get, reverse=True)
        vocab_list = vocab_list[:n_open]
        self.open_states = {st: 1 for st in vocab_list}

    def _init_params(self, X, y):
        """Initialize state space, obs space and priors."""
        self.states = set(st for state_seq in y for st in state_seq)
        self.obs = set(o for obs_seq in X for o in obs_seq)
        # NOTE: plus-one smoothing for all possible combinations.
        self.init_probs = {st: 1. for st in self.states}
        self.trans = {prev_st: {st: 1. for st in self.states} for prev_st in self.states}
        self.emiss = {st: {} for st in self.states}

    def _fit_params(self, X, y):
        """MAP estimation of HMM probability distributions."""
        for obs_seq, state_seq in zip(X, y):
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

    def _normalize_params(self):
        """All probability distributions must sum to one."""
        normalize_dict(self.init_probs)
        normalize_dict(self.open_states)

        for st in self.states:
            normalize_dict(self.trans[st])
            normalize_dict(self.emiss[st])

    def decode(self, X, likelihood=logaddexp):
        """Find the most likely (hidden) state seq given observation seq X,
        through the Viterbi decoding algorithm.
        """
        return [self.viterbi(obs_seq, likelihood=likelihood) for obs_seq in X]

    def viterbi(self, obs_seq, likelihood):
        """Viterbi decoding algorithm."""
        T = len(obs_seq)
        Pr = {t: {st: 0. for st in self.states} for t in range(T)}
        V = {t: {st: None for st in self.states} for t in range(T)}

        # Base case fwd
        init_obs = obs_seq[0]
        for st in self.states:
            if init_obs in self.emiss[st]:
                Pr[0][st] = likelihood(self.init_probs[st], self.emiss[st][init_obs])
            elif init_obs not in self.obs and st in self.open_states:
                # obs unseen during training, consider only open states.
                Pr[0][st] = likelihood(self.init_probs[st])
            else:
                Pr[0][st] = 0.

        self._forward(Pr, V, obs_seq, likelihood)

        best_state = max(Pr[T-1], key=Pr[T-1].get)
        best_path = self._backward(V, best_state, T)

        return best_path

    def _forward(self, Pr, V, obs_seq, likelihood):
        """Fwd pass of the viterbi decoding algorithm."""
        for t, obs in enumerate(obs_seq[1:], start=1):
            for st in self.states:
                if obs in self.emiss[st]:
                    emiss_prob = self.emiss[st][obs]
                    prev_max_st = self._compute_max_state(Pr, t, st, emiss_prob,
                                                          likelihood)
                elif obs not in self.obs and st in self.open_states:
                    # obs unseen during training, consider only open states.
                    emiss_prob = self.open_states[st]
                    prev_max_st = self._compute_max_state(Pr, t, st, emiss_prob,
                                                          likelihood)
                else:
                    Pr[t][st] = 0.
                if 0 != Pr[t][st]:
                    V[t][st] = prev_max_st

    def _backward(self, V, best_state, T):
        """Backtracking."""
        out_path = [best_state]
        try:
            for t in range(T-1, 0, -1):
                best_state = V[t][best_state]
                out_path.insert(0, best_state)
            return out_path
        except KeyError:
            print(f"KeyError, t={t}.")
            while t > 0:
                out_path.append(None)
                t -= 1
                assert len(out_path) == T
                return out_path

    def _compute_max_state(self, Pr, t, state, emiss_prob, likelihood):
        """Argmax."""
        max_state = None
        for prev_st in self.states:
            prob_so_far = Pr[t-1][prev_st]
            trans_prob = self.trans[prev_st][state]

            cur_prob = likelihood(prob_so_far, trans_prob, emiss_prob)
            if cur_prob > Pr[t][state]:
                Pr[t][state] = cur_prob
                max_state = prev_st

        return max_state

    def save_params(self, file_name="hmmmodel.txt"):
        """Write model params to human-interpretable, json-format txt file."""
        params = vars(self)  # NOTE: dangerous, doesn't create copy
        for attr_name, val in params.items():
            if isinstance(val, set):
                # NOTE: sets can't be converted to json, go figure
                params[attr_name] = list(val)

        json_txt = json.dumps(vars(self), indent=4)
        with open(file_name, mode='w') as f:
            f.write(json_txt)

    def load_params(self, file_name="hmmmodel.txt"):
        """Load model params from a json-formatted txt file."""
        with open(file_name, mode='r') as f:
            params = json.load(f)

        for attr_name, val in params.items():
            if isinstance(val, list):
                val = set(val)
            setattr(self, attr_name, val)

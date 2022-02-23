"""Hidden Markov model class."""
import json

from utils import normalize_dict, logaddexp2, logaddexp

class HMM:
    """Hidden Markov model with discrete (i.e. categorical) emissions."""
    def __init__(self,):
        self.states = None
        self.obs = None
        self.init_probs = None # assume categorical dist.
        self.trans = None      # assume categorical dist.
        self.emiss = None      # assume categorical dist.

    def fit(self, X, y):
        """Estimate HMM parameters."""
        self._init_params(X, y)
        self._fit_params(X, y)
        self._normalize_params()
    
    def _init_params(self, X, y):
        """Initialize state space, obs space and priors."""
        self.states = set(st for state_seq in y for st in state_seq)
        self.obs = set(o for obs_seq in X for o in obs_seq)
        # NOTE: plus-one smoothing for all possible combinations.
        self.init_probs = {st: 1 for st in self.states}
        self.trans = {st: {st: 1 for st in self.states} for st in self.states}
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
        # NOTE: n_seq is NOT equal to len(X) due to plus-one smoothing
        #n_seq = sum(self.init_probs.values())
        normalize_dict(self.init_probs)
        for st in self.states:
            normalize_dict(self.trans[st])
            normalize_dict(self.emiss[st])

    def decode(self, X, likelihood=logaddexp):
        """Find the most likely (hidden) state seq given observation seq X,
        through the Viterbi decoding algorithm.
        """
        return [self.viterbi(obs_seq, likelihood=likelihood) for obs_seq in X]

    def viterbi(self, obs_seq, likelihood=logaddexp):
        """Viterbi decoding algorithm.
        * Let Pr[t][s] denote the probability of ending up in state s
        at time t, given the most probable path s*[0:t-1].
        * Let V[t][s] denote the most likely prev state s*[t-1],
        of the optimal path s*[0:t-1], ending up in state s at timestep t.
        """
        T, init_obs = len(obs_seq), obs_seq[0]
        init_obs = obs_seq[0]
        # Base case fwd
        Pr = {t: {st: 0. for st in self.states} for t in range(T)}
        for st in self.states:
            if init_obs in self.emiss[st]:
                Pr[0][st] = likelihood(self.init_probs[st], self.emiss[st][init_obs])
            elif init_obs in self.obs:
                Pr[0][st] = 0.
            else:
                Pr[0][st] = likelihood(self.init_probs[st])

        V = {t: {st: None for st in self.states} for t in range(T)}

        # Forward pass
        for t, obs in enumerate(obs_seq[1:], start=1):
            for st in self.states:
                if obs in self.emiss[st]:
                    emiss_prob = self.emiss[st][obs]
                    prev_max_st = self._compute_max_state(Pr, t, st, emiss_prob, 
                                                          likelihood)
                elif obs in self.obs:
                    Pr[t][st] = 0.
                    prev_max_st = None
                else:  # unseen word
                    emiss_prob = 0
                    prev_max_st = self._compute_max_state(Pr, t, st, emiss_prob,
                                                          likelihood)
                    #Pr[t][st] = self.trans[st][prev_max_st]
                #prev_max_st = self._compute_max_state(Pr, t, st, emiss_prob)
                V[t][st] = prev_max_st

        # Backward pass
        best_state = max(Pr[T-1], key=Pr[T-1].get)
        out_path = [best_state]
        for t in range(T-1, 0, -1):
            try:
                best_state = V[t][best_state]
            except KeyError:
                print(t)
                return Pr, V
            out_path.insert(0, best_state)

        return out_path
    
    def _compute_max_state(self, Pr, t, state, emiss_prob, likelihood):
        max_state = None
        for prev_st in self.states:
            prob_so_far = Pr[t-1][prev_st]

            cur_prob = self._compute_likelihood(prob_so_far, state, prev_st, 
                                                emiss_prob, likelihood)
            if cur_prob > Pr[t][state]:
                Pr[t][state] = cur_prob
                max_state = prev_st

        return max_state

    def _compute_likelihood(self, prob_so_far, state, 
                            prev_st, emiss_prob, likelihood):
        trans_prob = self.trans[prev_st][state]
        return likelihood(prob_so_far, emiss_prob, trans_prob)

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
            #if "states" == key or "obs" == key:
            if isinstance(val, list):
                val = set(val)
            setattr(self, attr_name, val)

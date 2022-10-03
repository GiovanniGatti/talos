import random
from typing import Callable, Optional

import numpy as np

from talos import rmenv
from talos.continuing import history, env, rwd_fn, policy_state


def sample_transition_revenue(
        fare_sampler: Callable[[int], np.ndarray],
        cap: int,
        arrival_rate: float,
        price_sensitivity: float,
        fare_structure: np.ndarray,
        num_trials: int = 13_000_000) -> float:
    """
    This function provides a vectorized implementation for sampling a transition.
    :return: The sampled expectation of the transition reward
    """
    rand = np.random.default_rng()
    episodes = np.arange(num_trials)
    arrivals = rand.poisson(lam=arrival_rate, size=num_trials)
    arrivals_episode_indexes = np.repeat(episodes, arrivals)
    fares = fare_sampler(num_trials)
    purchase_prob = np.exp(-price_sensitivity *
                           (fare_structure[fares[arrivals_episode_indexes]] / np.min(fare_structure) - 1))
    wtp = rand.random(size=np.sum(arrivals)) < purchase_prob
    wtp_episode_indexes = arrivals_episode_indexes[wtp]
    wtp_episode_indexes, count_wtp = np.unique(wtp_episode_indexes, return_counts=True)
    bookings = np.clip(count_wtp, a_max=cap, a_min=0)
    sample_r = np.sum(fare_structure[fares[wtp_episode_indexes]] * bookings) / num_trials
    return sample_r.item()


class CapturingRwdFn(rwd_fn.RewardFn):

    def __init__(self, rwd: Optional[float] = None) -> None:
        self.rwd = rwd if rwd is not None else random.random()
        self.cap = np.array([])
        self.bookings = np.array([])
        self.selected_fares = np.array([])

    def __call__(self, cap: np.ndarray, bookings: np.ndarray, selected_fares: np.ndarray) -> float:
        self.cap = cap
        self.bookings = bookings
        self.selected_fares = selected_fares
        return self.rwd


class CapturingHistory(history.History):

    def __init__(self, nbins: Optional[int] = None):
        if nbins is None:
            nbins = random.randint(1, 10)
        self._nbins = nbins
        self.collected_data = []

    def store(self, data: np.ndarray) -> None:
        self.collected_data.append(data)

    def distribution(self) -> np.ndarray:
        return np.zeros(self._nbins)

    def nbins(self) -> int:
        return self._nbins

    def raw(self) -> np.ndarray:
        return np.array(self.collected_data)


class FixedDistributionHistory(history.History):

    def __init__(self, with_distribution: np.ndarray):
        self._with_distribution = with_distribution

    def store(self, data: np.ndarray) -> None:
        pass

    def distribution(self) -> np.ndarray:
        return self._with_distribution

    def nbins(self) -> int:
        return self._with_distribution.shape[0]

    def raw(self) -> np.ndarray:
        # this history type uses distribution only data
        raise NotImplementedError


class RawHistory(history.History):

    def __init__(self, _history: np.ndarray):
        self._nbins = _history[0].shape[0]
        self._history = _history

    def store(self, data: np.ndarray) -> None:
        pass

    def distribution(self) -> np.ndarray:
        return np.sum(self._history, axis=0)

    def raw(self) -> np.ndarray:
        return self._history

    def nbins(self) -> int:
        return self._nbins


def any_continuing_single_leg_env_with(_sampler: rmenv.DemandParameterState,
                                       horizon: Optional[int] = None,
                                       initial_capacity: Optional[int] = None,
                                       fare_structure: Optional[np.ndarray] = None,
                                       _policy_state: Optional[rmenv.PolicyState] = None,
                                       _rwd_fn: Optional[rwd_fn.RewardFn] = None) -> env.ContinuingSingleLeg:
    horizon = horizon if horizon else np.random.randint(2, 365)
    initial_capacity = initial_capacity if initial_capacity else np.random.randint(5, 50)
    fare_structure = fare_structure if fare_structure is not None else \
        np.sort(np.random.randint(50, 250, size=np.random.randint(1, 15)))
    _policy_state = _policy_state if _policy_state else \
        policy_state.OptimalPolicy(initial_capacity, horizon, fare_structure)
    _policy_state.update(_sampler.arrival_rate, _sampler.price_sensitivity)
    _rwd_fn = _rwd_fn if _rwd_fn else CapturingRwdFn()
    return env.ContinuingSingleLeg(
        initial_capacity, horizon, fare_structure, _sampler, _policy_state, _rwd_fn)

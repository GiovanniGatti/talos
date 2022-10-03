import abc
from functools import lru_cache

import numpy as np
from scipy.special import gammaincc, gammainc, gamma

from talos import rmenv


class RewardFn(abc.ABC):

    @abc.abstractmethod
    def __call__(self, cap: np.ndarray, bookings: np.ndarray, selected_fares: np.ndarray) -> float:
        pass


class ExpectedRwd(RewardFn):

    def __init__(self, initial_cap: int, fare_structure: np.ndarray, param_state: rmenv.DemandParameterState):
        self._fare_structure = fare_structure
        self._param_state = param_state
        self._bkgs = np.arange(1, initial_cap + 1)
        self._f0 = np.min(fare_structure).item()

    @lru_cache(maxsize=256)
    def _expected_rwd(self, arrival_rate: float, price_sensitivity: float) -> np.ndarray:
        mu = arrival_rate * np.exp(-price_sensitivity * (self._fare_structure / self._f0 - 1))
        mu = np.broadcast_to(mu, (self._bkgs.shape[0], mu.shape[0]))
        expected_bookings = (mu * gammaincc(self._bkgs, mu.T).T
                             - ((np.exp(-mu) * (mu.T ** self._bkgs).T).T / gamma(self._bkgs)).T
                             + (self._bkgs * gammainc(self._bkgs, mu.T)).T)
        expected_rwd = expected_bookings * self._fare_structure
        expected_rwd = np.concatenate((np.zeros((1, self._fare_structure.shape[0])), expected_rwd))
        return expected_rwd

    def __call__(self, cap: np.ndarray, bookings: np.ndarray, selected_fares: np.ndarray) -> float:
        expected_rwd = self._expected_rwd(self._param_state.arrival_rate, self._param_state.price_sensitivity)
        exp_rwd: float = np.sum(expected_rwd[np.array(cap), selected_fares]).item()
        return exp_rwd


class StochasticRwd(RewardFn):

    def __init__(self, fare_structure: np.ndarray):
        self._fare_structure = fare_structure

    def __call__(self, cap: np.ndarray, bookings: np.ndarray, selected_fares: np.ndarray) -> float:
        rwd: float = np.sum(bookings * self._fare_structure[selected_fares]).item()
        return rwd

import itertools
import random
from collections import defaultdict
from functools import lru_cache
from typing import Tuple, Callable, Union

import numpy as np
from scipy import integrate
from scipy.stats import truncnorm

from talos import dp
from talos import rmenv


@lru_cache(maxsize=16)
def dynamic_programming(
        booking_horizon: int, leg_cap: int, arrival_rate: float, price_sensitivity: float, fare_structure: Tuple[float]
) -> Tuple[np.ndarray, np.ndarray]:
    q = dp.time_constant_dynamic_programming(
        leg_cap, arrival_rate, price_sensitivity, np.array(fare_structure), booking_horizon)
    v = np.max(q, axis=-1)
    v = np.concatenate((np.zeros_like(v[0, :]).reshape(1, -1), v))
    policy = q.argmax(axis=-1)
    policy = np.concatenate((np.zeros_like(policy[0, :]).reshape(1, -1), policy))
    return policy, v


@lru_cache(maxsize=16)
def average_policy(booking_horizon: int,
                   leg_cap: int,
                   arrival_rate: float,
                   price_sensitivity: float,
                   fare_structure: Tuple[float],
                   price_sensitivity_std: float,
                   min_ps: float,
                   max_ps: float,
                   k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    assert price_sensitivity_std >= 0.
    assert min_ps < max_ps
    fare_structure = np.array(fare_structure)

    if price_sensitivity_std < 1e-5:  # avoid zero std
        _q_table = dp.time_constant_dynamic_programming(
            leg_cap, arrival_rate, price_sensitivity, fare_structure, booking_horizon)
        return _q_table.argmax(axis=-1), np.max(_q_table, axis=-1)

    avg_q = np.zeros((booking_horizon, leg_cap + 1, len(fare_structure)))
    policy = avg_q.argmax(axis=-1)

    # For performance reasons, this function has been copied from talos.dp and improved locally
    @lru_cache(maxsize=None)
    def _transition_probability_matrix(
            _price_sensitivity: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_bkgs = dp.dp._max_bkgs(arrival_rate)

        purchase_prob = np.exp(-_price_sensitivity * (fare_structure / np.min(fare_structure) - 1))
        purchase_prob = purchase_prob[:, None]

        bkgs = np.arange(1, max_bkgs + 1)
        ln_fac = np.cumsum(np.log(bkgs))

        bkgs = np.tile(bkgs, (purchase_prob.shape[0], 1))
        ln_purchase_prob = bkgs * np.log(purchase_prob)
        ln_arrival_rate = bkgs * np.log(arrival_rate)
        transition_probabilities = ln_arrival_rate + ln_purchase_prob - arrival_rate * purchase_prob - ln_fac
        transition_probabilities = np.concatenate(((- arrival_rate * purchase_prob), transition_probabilities), axis=1)
        transition_probabilities = np.exp(transition_probabilities)

        purchase_prob = np.exp(-_price_sensitivity * (fare_structure / np.min(fare_structure) - 1))
        purchase_prob = purchase_prob[:, None]

        bkgs = np.arange(1, max_bkgs + 1)
        bkgs = np.tile(bkgs, (purchase_prob.shape[0], 1))
        bkgs = np.tile(bkgs, (max_bkgs, 1, 1))
        capacity_constraint = np.repeat(max_bkgs - np.arange(max_bkgs), bkgs.shape[1] * bkgs.shape[2]) \
            .reshape((max_bkgs, fare_structure.shape[0], max_bkgs))
        np.clip(bkgs, np.zeros_like(bkgs), capacity_constraint, out=bkgs)
        bkgs = np.flip(bkgs, axis=0)
        exp_reward = np.sum(bkgs * fare_structure[:, None] * transition_probabilities[:, 1:], axis=2)
        exp_reward = np.concatenate((np.zeros((1, exp_reward.shape[1])), exp_reward))
        transition_probabilities = np.flipud(np.rot90(transition_probabilities))  # 2-d vector (bkgs x fare)

        indexes = np.arange(leg_cap + 1)[:, None] - np.tile(np.arange(0, max_bkgs + 1), (leg_cap + 1, 1))
        np.clip(indexes, 0, leg_cap, out=indexes)
        rwd_indexes = np.clip(indexes[:, 0], 0, max_bkgs - 1)

        return transition_probabilities, exp_reward, indexes, rwd_indexes

    _min = max(min_ps, price_sensitivity - 5. * price_sensitivity_std)
    _max = min(max_ps, price_sensitivity + 5. * price_sensitivity_std)

    x = np.linspace(_min, _max, num=2 ** k + 1)
    dx = x[1] - x[0]

    a = (_min - price_sensitivity) / price_sensitivity_std
    b = (_max - price_sensitivity) / price_sensitivity_std

    @lru_cache(maxsize=None)
    def _pdf(ps: float) -> float:
        return truncnorm.pdf(ps, a, b, loc=price_sensitivity, scale=price_sensitivity_std)

    point_q = defaultdict(lambda: np.zeros((booking_horizon, leg_cap + 1, fare_structure.shape[0])))

    states = np.arange(leg_cap + 1)
    for _dtd, _cap in itertools.product(range(booking_horizon), range(1, leg_cap + 1)):
        actions = policy[_dtd - 1]
        for ps in x:
            q = point_q[ps]
            transition_probabilities, exp_reward, indexes, rwd_indexes = _transition_probability_matrix(ps)
            _v = q[_dtd - 1, states, actions]
            q[_dtd] = exp_reward[rwd_indexes] + np.dot(_v[indexes], transition_probabilities)
        for _fare in range(fare_structure.shape[0]):
            samples = [_pdf(ps) * point_q[ps][_dtd, _cap, _fare] for ps in x]
            avg_q[_dtd, _cap, _fare] = integrate.romb(samples, dx=dx)
        policy = avg_q.argmax(axis=-1)

    v = np.max(avg_q, axis=-1)
    return policy, v


@lru_cache(maxsize=16)
def _random_eval(
        booking_horizon: int, leg_cap: int, arrival_rate: float, price_sensitivity: float, fare_structure: Tuple[float]
) -> Tuple[np.ndarray, np.ndarray]:
    q = dp.time_constant_random_policy_evaluation(
        leg_cap, arrival_rate, price_sensitivity, np.array(fare_structure), booking_horizon)
    v = np.mean(q, axis=-1)
    v = np.concatenate((np.zeros_like(v[0, :]).reshape(1, -1), v))
    return np.array([]), v


class _BasePolicy(rmenv.PolicyState):

    def __init__(self,
                 cap: int,
                 horizon: int,
                 fare_structure: Union[Tuple[float], np.ndarray],
                 policy_fn: Callable[[int, int, float, float, Tuple[float]], Tuple[np.ndarray, np.ndarray]]):
        super().__init__()
        self._cap = cap
        self._fare_structure = tuple(fare_structure)
        self._horizon = horizon
        self._policy_fn = policy_fn
        self._v = None
        self._pi = None

    def horizon(self) -> int:
        return self._horizon

    def initial_cap(self) -> int:
        return self._cap

    def v(self, dtd: int, cap: int) -> float:
        assert self._v is not None
        assert 0 <= dtd <= self._horizon and 0 <= cap <= self._cap
        return self._v[dtd, cap].item()

    def pi(self, dtd: int, cap: int) -> float:
        assert self._pi is not None
        assert 0 <= dtd <= self._horizon and 0 <= cap <= self._cap
        return self._pi[dtd, cap].item()

    def __getitem__(self, item: Tuple[np.ndarray, ...]) -> np.ndarray:
        assert self._pi is not None
        return self._pi[item]

    def update(self, new_arrival_rate: float, new_price_sensitivity: float) -> None:
        self._pi, self._v = self._policy_fn(
            self._horizon, self._cap, new_arrival_rate, new_price_sensitivity, self._fare_structure)


class OptimalPolicy(_BasePolicy):

    def __init__(self, cap: int, horizon: int, fare_structure: Union[Tuple[float], np.ndarray]):
        super().__init__(cap, horizon, fare_structure, dynamic_programming)


class RandomPolicy(_BasePolicy):

    def __init__(self, cap: int, horizon: int, fare_structure: Union[Tuple[float], np.ndarray]):
        super().__init__(cap, horizon, fare_structure, _random_eval)
        self._rand = random.Random()
        self._rng = np.random.default_rng()

    def pi(self, dtd: int, cap: int) -> float:
        return self._rand.randint(0, len(self._fare_structure) - 1)

    def __getitem__(self, item: Tuple[np.ndarray, ...]) -> np.ndarray:
        all_len = set(map(lambda a: a.shape, item))
        assert len(all_len) == 1
        input_len = next(iter(all_len))
        assert len(input_len) == 1
        return self._rng.integers(len(self._fare_structure), size=input_len)

from typing import Callable

import numpy as np
from scipy import stats
from scipy.special import gammainc


def time_constant_dynamic_programming(
        leg_cap: int, arrival_rate: float, price_sensitivity: float, fare_structure: np.ndarray, horizon: int
) -> np.ndarray:
    """
     Speed up implementation of dynamic programming use numpy vectorization at its best. It improves speed about 100x
     with respect generic implementation of DP provided at dynamic_programming method. Note that this implementation
     assumes constant arrival rate and constant purchase probability on a single leg scenario.
     """
    return time_constant_policy_evaluation(
        leg_cap, arrival_rate, price_sensitivity, fare_structure, horizon, lambda dtd, q: np.max(q, axis=-1))


def time_constant_random_policy_evaluation(
        leg_cap: int, arrival_rate: float, price_sensitivity: float, fare_structure: np.ndarray, horizon: int
) -> np.ndarray:
    """
     Returns the Q-Table for a random policy
     """
    return time_constant_policy_evaluation(
        leg_cap, arrival_rate, price_sensitivity, fare_structure, horizon, lambda dtd, q: np.mean(q, axis=-1))


def time_constant_policy_evaluation(leg_cap: int,
                                    arrival_rate: float,
                                    price_sensitivity: float,
                                    fare_structure: np.ndarray,
                                    horizon: int,
                                    bootstrapping_fn: Callable[[int, np.ndarray], np.ndarray]) -> np.ndarray:
    if price_sensitivity <= 0.:
        raise ValueError('Expected a positive price sensitivity')

    if arrival_rate <= 0.:
        raise ValueError('Expected a positive arrival rate')

    q = np.zeros((horizon, leg_cap + 1, fare_structure.shape[0]))
    transition_probabilities = transition_probability_matrix(arrival_rate, price_sensitivity, fare_structure)

    max_bkgs = transition_probabilities.shape[1] - 1

    purchase_prob = np.exp(-price_sensitivity * (fare_structure / np.min(fare_structure) - 1))
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

    for dtd in range(horizon):
        v = bootstrapping_fn(dtd, q[dtd - 1])
        q[dtd] = exp_reward[rwd_indexes] + np.dot(v[indexes], transition_probabilities)

    return q


def transition_probability_matrix(
        arrival_rate: float, price_sensitivity: float, fare_structure: np.ndarray
) -> np.ndarray:
    """
    :returns transition probabilities for unconstrained capacity (fares x #bkgs)
    """
    max_bkgs = _max_bkgs(arrival_rate)

    purchase_prob = np.exp(-price_sensitivity * (fare_structure / np.min(fare_structure) - 1))
    purchase_prob = purchase_prob[:, None]

    bkgs = np.arange(1, max_bkgs + 1)
    ln_fac = np.cumsum(np.log(bkgs))

    bkgs = np.tile(bkgs, (purchase_prob.shape[0], 1))
    ln_purchase_prob = bkgs * np.log(purchase_prob)
    ln_arrival_rate = bkgs * np.log(arrival_rate)
    transition_probabilities = ln_arrival_rate + ln_purchase_prob - arrival_rate * purchase_prob - ln_fac
    transition_probabilities = np.concatenate(((- arrival_rate * purchase_prob), transition_probabilities), axis=1)
    return np.exp(transition_probabilities)


def state_space_distribution(leg_cap: int,
                             arrival_rate: float,
                             price_sensitivity: float,
                             fare_structure: np.ndarray,
                             horizon: int,
                             policy: np.ndarray) -> np.ndarray:
    if price_sensitivity <= 0.:
        raise ValueError('Expected a positive price sensitivity')

    if arrival_rate <= 0.:
        raise ValueError('Expected a positive arrival rate')

    transition_probabilities = transition_probability_matrix(arrival_rate, price_sensitivity, fare_structure)

    max_bkgs = transition_probabilities.shape[1] - 1
    mu = arrival_rate * np.exp(-price_sensitivity * (fare_structure / np.min(fare_structure) - 1))

    constrained_transition_probabilities = gammainc(
        np.arange(max_bkgs + 1), np.repeat(mu, max_bkgs + 1).reshape(fare_structure.shape[0], -1))

    _max_t = min(max_bkgs + 1, leg_cap + 1)
    states = np.arange(leg_cap + 1)
    constrained_probs_indexes = np.clip(states, 0, _max_t - 1)
    probs_indexes = np.repeat(states, leg_cap + 1).reshape(leg_cap + 1, -1)
    indexes = states[:, None] - np.tile(np.arange(0, leg_cap + 1)[::-1], (leg_cap + 1, 1))
    mask = np.logical_and(0 <= indexes, indexes < _max_t)
    indexes[np.logical_not(mask)] = -1
    to_replace_idx = np.argmax(mask, axis=1) + constrained_probs_indexes

    distr = np.zeros(shape=(horizon, leg_cap + 1))
    distr[-1, -1] = 1.
    for dtd in reversed(range(1, horizon)):
        probs = transition_probabilities[policy[dtd]]
        constrained_probs = constrained_transition_probabilities[policy[dtd]]
        t = probs[probs_indexes, indexes]
        t[states, to_replace_idx] = constrained_probs[constrained_probs_indexes, constrained_probs_indexes]
        distr[dtd - 1] = np.dot(distr[dtd], mask * t)[::-1]

    return distr


def fare_class_distribution(leg_cap: int,
                            arrival_rate: float,
                            price_sensitivity: float,
                            fare_structure: np.ndarray,
                            horizon: int,
                            policy: np.ndarray) -> np.ndarray:
    if price_sensitivity <= 0.:
        raise ValueError('Expected a positive price sensitivity')

    if arrival_rate <= 0.:
        raise ValueError('Expected a positive arrival rate')

    distr = state_space_distribution(leg_cap, arrival_rate, price_sensitivity, fare_structure, horizon, policy)
    distr = distr[:, 1:]
    policy = policy[:, 1:]

    stacked = np.tile(policy[np.newaxis], (10, 1, 1))
    stacked_distr = np.tile(distr[np.newaxis], (10, 1, 1))
    size = fare_structure.shape[0]
    seq = np.repeat(np.arange(size), np.prod(policy.shape)).reshape((size,) + policy.shape)
    class_weight = np.sum(stacked_distr * (stacked == seq), axis=(1, 2))
    return class_weight / np.sum(class_weight)


def _max_bkgs(arrival_rate: float, cdf_upper_bound: float = 10 ** -5) -> int:
    max_bkgs = 0
    while True:
        cdf = stats.poisson.cdf(max_bkgs, arrival_rate)
        if 1. - cdf <= cdf_upper_bound:
            break
        max_bkgs += 1
    return max_bkgs

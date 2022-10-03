import math
import random
import unittest

import numpy as np
import pytest
from scipy.stats import truncnorm

from talos import dp
from talos.continuing import policy_state
from tests.talos.continuing import tools


class TestOptimalPolicy(unittest.TestCase):

    def test_raises_assertion_error_if_value_has_not_been_updated_before_query(self) -> None:
        opt_policy = policy_state.OptimalPolicy(cap=50, horizon=365, fare_structure=np.array([50., 70.]))

        with pytest.raises(AssertionError):
            opt_policy.v(0, 0)

    def test_raises_assertion_error_if_policy_has_not_been_updated_before_query(self) -> None:
        opt_policy = policy_state.OptimalPolicy(cap=50, horizon=365, fare_structure=np.array([50., 70.]))

        with pytest.raises(AssertionError):
            opt_policy.pi(0, 0)

        with pytest.raises(AssertionError):
            _ = opt_policy[np.array([0]), np.array([0])]

    def test_computed_v_is_close_to_sampled_v(self) -> None:
        fare_structure = np.arange(50., 250., 20.)
        opt_policy = policy_state.OptimalPolicy(cap=50, horizon=365, fare_structure=fare_structure)
        any_arrival_rate = random.randint(60, 100) / 365
        any_remaining_cap = random.randint(1, 3)
        any_price_sensitivity = math.log(2.) / (random.uniform(1.9, 3.3) - 1.)
        opt_policy.update(any_arrival_rate, any_price_sensitivity)
        fare = opt_policy.pi(dtd=1, cap=any_remaining_cap)

        v = opt_policy.v(1, any_remaining_cap)
        sampled_v = tools.sample_transition_revenue(
            lambda i: np.repeat(fare, i), any_remaining_cap, any_arrival_rate, any_price_sensitivity, fare_structure)

        assert pytest.approx(sampled_v, rel=0.01) == v

    def test_pi_returns_optimal_fare(self) -> None:
        fare_structure = np.arange(50., 250., 20.)
        opt_policy = policy_state.OptimalPolicy(cap=50, horizon=365, fare_structure=fare_structure)
        arrival_rate = 80 / 365
        price_sensitivity = math.log(2.) / (1.75 - 1.)
        opt_policy.update(arrival_rate, price_sensitivity)
        assert opt_policy.pi(365, 1) == np.argmax(fare_structure)
        assert opt_policy.pi(1, 50) == (np.abs(fare_structure - 50. / price_sensitivity)).argmin()


class TestRandomPolicy(unittest.TestCase):

    def test_raises_assertion_error_if_policy_has_not_been_updated_before_query(self) -> None:
        rnd_policy = policy_state.RandomPolicy(cap=50, horizon=365, fare_structure=np.array([50., 70.]))

        with pytest.raises(AssertionError):
            rnd_policy.v(0, 0)

    def test_computed_v_is_close_to_sampled_v(self) -> None:
        fare_structure = np.arange(50., 250., 20.)
        rnd_policy = policy_state.RandomPolicy(cap=50, horizon=365, fare_structure=fare_structure)
        any_arrival_rate = random.randint(60, 100) / 365
        any_remaining_cap = random.randint(1, 3)
        any_price_sensitivity = math.log(2.) / (random.uniform(1.9, 3.3) - 1.)
        rnd_policy.update(any_arrival_rate, any_price_sensitivity)

        v = rnd_policy.v(1, any_remaining_cap)
        sampled_v = tools.sample_transition_revenue(
            lambda i: np.random.randint(fare_structure.shape[0], size=i),
            any_remaining_cap, any_arrival_rate, any_price_sensitivity, fare_structure)

        assert pytest.approx(sampled_v, rel=0.01) == v

    def test_pi_returns_random_fare(self) -> None:
        fare_structure = np.arange(50., 250., 20.)
        rdn_policy = policy_state.RandomPolicy(cap=50, horizon=365, fare_structure=fare_structure)
        any_arrival_rate = random.randint(60, 100) / 365
        any_remaining_cap = random.randint(1, 3)
        rdn_policy.update(any_arrival_rate, any_remaining_cap)

        collected = []
        for _ in range(50_000):
            f = rdn_policy.pi(3, 5)
            collected.append(f)
        _, counts = np.unique(collected, return_counts=True)

        assert (np.isclose(counts / 50_000, np.repeat(0.1, counts.shape[0]), atol=0.005)).all()


class TestAvgPolicy(unittest.TestCase):

    def test_return_optimal_policy_when_small_uncertainty(self) -> None:
        pi, _ = policy_state.average_policy(booking_horizon=22,
                                            leg_cap=6,
                                            arrival_rate=7. / 22,
                                            price_sensitivity=math.log(2) / (2.75 - 1),
                                            fare_structure=tuple(range(50, 250, 20)),
                                            price_sensitivity_std=0.001,
                                            min_ps=math.log(2) / (4.05 - 1),
                                            max_ps=math.log(2) / (1.05 - 1))

        opt_pi, _ = policy_state.dynamic_programming(booking_horizon=22,
                                                     leg_cap=6,
                                                     arrival_rate=7. / 22,
                                                     price_sensitivity=math.log(2) / (2.75 - 1),
                                                     fare_structure=tuple(range(50, 250, 20)))

        assert np.all(pi == opt_pi[1:]), 'Expected average policy to have same behavior as optimal ' \
                                         'policy when uncertainty is small'

    def test_value_function_correctly_predicts_expected_revenue(self) -> None:
        pi, v = policy_state.average_policy(booking_horizon=22,
                                            leg_cap=6,
                                            arrival_rate=7. / 22,
                                            price_sensitivity=math.log(2) / (2.75 - 1),
                                            fare_structure=tuple(range(50, 250, 20)),
                                            price_sensitivity_std=.15,
                                            min_ps=math.log(2) / (4.05 - 1),
                                            max_ps=math.log(2) / (1.05 - 1))

        a = (math.log(2) / (4.05 - 1) - math.log(2) / (2.75 - 1)) / .15
        b = (math.log(2) / (1.05 - 1) - math.log(2) / (2.75 - 1)) / .15

        samples = truncnorm.rvs(a, b, loc=math.log(2) / (2.75 - 1), scale=.15, size=1000)

        total = 0.
        for n, sample in enumerate(samples):
            q = dp.time_constant_policy_evaluation(
                horizon=22,
                leg_cap=6,
                arrival_rate=7. / 22,
                price_sensitivity=sample,
                fare_structure=np.arange(50, 250, 20),
                bootstrapping_fn=lambda dtd, _q: _q[np.arange(7), pi[dtd - 1]])
            total += (q[-1, -1, pi[-1, -1]] - total) / (n + 1)

        assert pytest.approx(v[-1, -1], rel=0.015) == total


if __name__ == '__main__':
    unittest.main()

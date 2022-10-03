import math
import random
import unittest

import numpy as np
import pytest
from scipy.stats import poisson

from talos.continuing import rwd_fn
from talos.rmenv import DemandParameterState


class TestExpRwd(unittest.TestCase):

    def test_returns_expectation_under_unconstrained_capacity(self) -> None:
        param_state = DemandParameterState(arrival_rate=.6, price_sensitivity=.3)
        _rwd_fn = rwd_fn.ExpectedRwd(initial_cap=random.randint(100, 150),
                                     fare_structure=np.array([50., 70.]),
                                     param_state=param_state)
        any_amount_of_bookings = np.random.randint(0, 10, size=(2,))

        rwd = _rwd_fn(cap=np.array([100, 100]), bookings=any_amount_of_bookings, selected_fares=np.array([0, 1]))

        assert rwd == 50. * 0.6 + 70 * 0.6 * math.exp(-0.3 * (70. / 50. - 1))

    def test_returns_expectation_under_extremely_likelihood_of_selling_all_remaining_places(self) -> None:
        param_state = DemandParameterState(arrival_rate=100., price_sensitivity=.3)
        _rwd_fn = rwd_fn.ExpectedRwd(initial_cap=random.randint(100, 150),
                                     fare_structure=np.array([50., 70.]),
                                     param_state=param_state)
        any_amount_of_bookings = np.random.randint(0, 10, size=(2,))

        rwd = _rwd_fn(cap=np.array([1, 1]), bookings=any_amount_of_bookings, selected_fares=np.array([0, 1]))

        assert rwd == 50. + 70.

    def test_returns_expectation_under_constrained_capacity(self) -> None:
        param_state = DemandParameterState(arrival_rate=.6, price_sensitivity=.3)
        _rwd_fn = rwd_fn.ExpectedRwd(initial_cap=random.randint(100, 150),
                                     fare_structure=np.array([50., 70.]),
                                     param_state=param_state)
        any_amount_of_bookings = np.random.randint(0, 10, size=(1,))

        rwd = _rwd_fn(cap=np.array([2]), bookings=any_amount_of_bookings, selected_fares=np.array([1]))

        _mu = .6 * math.exp(-0.3 * (70. / 50. - 1))
        assert rwd == pytest.approx(70. * poisson.pmf(k=1, mu=_mu) + 2 * 70. * (1 - poisson.cdf(k=1, mu=_mu)),
                                    abs=1e-12)


class TestStochasticRwd(unittest.TestCase):

    def test_returns_the_sum_of_bookings_times_selected_fares(self) -> None:
        _rwd_fn = rwd_fn.StochasticRwd(fare_structure=np.array([50., 70., 90.]))
        any_cap = np.random.randint(0, 10, size=(3,))

        rwd = _rwd_fn(any_cap, bookings=np.array([1, 0, 2]), selected_fares=np.array([0, 1, 2]))
        assert rwd == 1 * 50. + 0 * 70. + 2 * 90.

        rwd = _rwd_fn(any_cap, bookings=np.array([0, 2, 1]), selected_fares=np.array([0, 1, 2]))
        assert rwd == 0 * 50. + 2 * 70. + 1 * 90.

        rwd = _rwd_fn(any_cap, bookings=np.array([0, 2, 1]), selected_fares=np.array([2, 0, 1]))
        assert rwd == 0 * 90. + 2 * 50. + 1 * 70.


if __name__ == '__main__':
    unittest.main()

import unittest
from typing import List, Any, Dict, Tuple

import gym
import numpy as np
import pytest

from talos.continuing import history
from tests.talos import continuing


class _InfoEnv(gym.Env):

    def __init__(self, with_info: List[Dict[Any, Any]]):
        self._info = iter(with_info)

    def step(self, action) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        return 0, 1., False, next(self._info)

    def reset(self) -> Any:
        return 0

    def render(self, mode="human"):
        pass


class TestSumHistory(unittest.TestCase):

    def test_history_keeps_track_of_past_data(self) -> None:
        _history = history.SumHistory(3, horizon=3)

        for data in [np.array([0, 2, 1]),
                     np.array([1, 1, 2]),
                     np.array([1, 0, 2])]:
            _history.store(data)

        assert np.alltrue(_history.distribution() == np.array([2, 3, 5]))

    def test_history_discards_old_data(self) -> None:
        _history = history.SumHistory(3, horizon=2)

        for data in [np.array([0, 2, 1]),  # discarded
                     np.array([1, 1, 2]),
                     np.array([1, 0, 2])]:
            _history.store(data)

        assert np.alltrue(_history.distribution() == np.array([2, 1, 4]))


class TestBookingHistoryWrapper:

    @pytest.mark.parametrize(
        'daily_bkgs, action, expected',
        [(np.array([1, 2, 3]), np.array([0, 1, 2]), np.array([1, 2, 3])),
         (np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([0, 6, 0])),
         (np.array([4, 5, 6]), np.array([0, 1, 0]), np.array([10, 5, 0])),
         (np.array([1, 0, 1]), np.array([0, 1, 0]), np.array([2, 0, 0])),
         (np.array([1, 0, 1]), np.array([2, 1, 0]), np.array([1, 0, 1])),
         (np.array([1, 0, 0]), np.array([2, 1, 0]), np.array([0, 0, 1])),
         (np.array([0, 0, 0]), np.array([0, 1, 2]), np.array([0, 0, 0])), ]
    )
    def test_bkg_hist_wrapper_stores_daily_bookings_by_fare_class(
            self, daily_bkgs: np.ndarray, action: np.ndarray, expected: np.ndarray
    ) -> None:
        _env = _InfoEnv(with_info=[{'daily_bkgs': daily_bkgs}, ])
        capturing_history = continuing.CapturingHistory(nbins=3)  # number of fare classes
        wrapped = history.BookingHistoryWrapper(_env, capturing_history)

        wrapped.reset()
        wrapped.step(action)

        assert np.alltrue(capturing_history.collected_data == expected)

    def test_bkg_hist_wrapper_keeps_track_of_history_during_reset(self) -> None:
        _env = _InfoEnv(with_info=[{'daily_bkgs': np.array([7, 8, 9])}, ])

        def _reset_stub() -> Any:
            o, _, _, _ = _env.step(np.array([0, 1, 2]))
            return o

        _env.reset = _reset_stub

        capturing_history = continuing.CapturingHistory(nbins=3)  # number of fare classes
        wrapped = history.BookingHistoryWrapper(_env, capturing_history)

        wrapped.reset()

        assert np.alltrue(capturing_history.collected_data == np.array([[7, 8, 9]]))


class TestOfferHistoryWrapper:

    @pytest.mark.parametrize(
        'incomplete_flights, action, expected',
        [(np.array([True, True, True]), np.array([0, 1, 2]), np.array([1, 1, 1])),
         (np.array([True, True, True]), np.array([1, 1, 1]), np.array([0, 3, 0])),
         (np.array([False, False, True]), np.array([1, 1, 1]), np.array([0, 1, 0])),
         (np.array([False, False, False]), np.array([1, 1, 1]), np.array([0, 0, 0])), ]
    )
    def test_offer_hist_wrapper_stores_daily_offers_by_fare_class(
            self, incomplete_flights: np.ndarray, action: np.ndarray, expected: np.ndarray
    ) -> None:
        _env = _InfoEnv(with_info=[{'incomplete_flights': incomplete_flights}, ])
        capturing_history = continuing.CapturingHistory(nbins=3)  # number of fare classes
        wrapped = history.OfferHistoryWrapper(_env, capturing_history)

        wrapped.reset()
        wrapped.step(action)

        assert np.alltrue(capturing_history.collected_data == expected)

    def test_offer_hist_wrapper_keeps_track_of_history_during_reset(self) -> None:
        _env = _InfoEnv(with_info=[{'incomplete_flights': np.array([True, True, True])}, ])

        def _reset_stub() -> Any:
            o, _, _, _ = _env.step(np.array([1, 1, 1]))
            return o

        _env.reset = _reset_stub

        capturing_history = continuing.CapturingHistory(nbins=3)  # number of fare classes
        wrapped = history.OfferHistoryWrapper(_env, capturing_history)

        wrapped.reset()

        assert np.alltrue(capturing_history.collected_data == np.array([[0, 3, 0]]))


if __name__ == '__main__':
    unittest.main()

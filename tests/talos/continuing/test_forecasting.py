import random
import unittest
from typing import Tuple, Dict, Optional, List

import numpy as np
import pytest
from gym import spaces

from talos.continuing import forecasting
from talos.rmenv import DemandParameterState
from tests.talos import continuing, rmenv


class _DummyEstimator(forecasting.EstimatorState):

    @staticmethod
    def any_estimator_with(n_params: Optional[int] = None,
                           estimation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           bounds: Optional[np.ndarray] = None,
                           param_names: Optional[List[str]] = None) -> '_DummyEstimator':
        n_params = n_params if n_params is not None else random.randint(1, 5)
        bounds = bounds if bounds is not None else np.random.random() + np.arange(2 * n_params).reshape(-1, 2)
        estimation = estimation if estimation is not None else (
            np.random.uniform(bounds[:, 0], bounds[:, 1]), np.random.uniform(0, 2., size=2 * n_params).reshape(-1, 2))
        param_names = param_names if param_names is not None else ['dummy_{}'.format(i) for i in range(n_params)]
        return _DummyEstimator(n_params, estimation, bounds, param_names)

    def __init__(
            self, n_params: int, estimation: Tuple[np.ndarray, np.ndarray], bounds: np.ndarray, param_names: List[str]):
        self._n_params = n_params
        self._estimation = estimation
        self._bounds = bounds
        self._param_names = param_names

    def bounds(self) -> np.ndarray:
        return self._bounds

    def n_params(self) -> int:
        return self._n_params

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._estimation

    def params(self) -> Dict[str, np.ndarray]:
        return {n: self._estimation[0][i] for i, n in zip(range(self._n_params), self._param_names)}

    def std(self) -> Dict[str, np.ndarray]:
        return {n: self._estimation[1][i] for i, n in zip(range(self._n_params), self._param_names)}


class TestPriceSensitivityEstimator(unittest.TestCase):

    def test_price_sensitivity_is_estimated_correctly(self) -> None:
        # d($230) = [(8 / 22.) * exp(-0.4 * (230 / 50 - 1))] * 50000 trials ~ 4307.7
        offer_history = continuing.FixedDistributionHistory(
            with_distribution=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 50000]))
        bkg_history = continuing.FixedDistributionHistory(with_distribution=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 4307]))

        fare_structure = np.arange(50, 250, 20)
        param_state = DemandParameterState(arrival_rate=8 / 22., price_sensitivity=random.random())
        estimator = forecasting.PriceSensitivityEstimator(
            offer_history, bkg_history, fare_structure, param_state, bound=(0.25, 1.2))

        price_sensitivity, std = estimator.estimate()

        assert price_sensitivity == pytest.approx(0.4, abs=0.002)
        assert std == pytest.approx(0., abs=0.005)


class TestTwoParameterEstimator(unittest.TestCase):

    def test_arrival_rate_and_price_sensitivity_is_estimated_correctly(self) -> None:
        # d($50) = [(8 / 22.) * exp(-0.4 * (50 / 50 - 1))] * 50000 trials ~ 18181
        # d($230) = [(8 / 22.) * exp(-0.4 * (230 / 50 - 1))] * 50000 trials ~ 4307.7
        offer_history = continuing.FixedDistributionHistory(
            with_distribution=np.array([50000, 0, 0, 0, 0, 0, 0, 0, 0, 50000]))
        bkg_history = continuing.FixedDistributionHistory(
            with_distribution=np.array([18181, 0, 0, 0, 0, 0, 0, 0, 0, 4307]))

        fare_structure = np.arange(50, 250, 20)
        estimator = forecasting.TwoParamsEstimator(
            offer_history, bkg_history, fare_structure, bounds=[(0.2, 0.5), (0.25, 1.2)])

        params, std = estimator.estimate()

        arrival_rate, price_sensitivity = params
        arrival_rate_std, price_sensitivity_std = std

        assert arrival_rate == pytest.approx(8 / 22., abs=0.002)
        assert price_sensitivity == pytest.approx(0.4, abs=0.002)
        assert arrival_rate_std == pytest.approx(0., abs=0.005)
        assert price_sensitivity_std == pytest.approx(0., abs=0.005)


class TestForecastingWrapper:

    @pytest.mark.parametrize(
        'value, expected',
        [(.1, -1.),
         (.2, 0.),
         (.3, 1.), ])
    def test_normalized_param_is_appended_to_observation(self, value: float, expected: float) -> None:
        env = rmenv.DummyEnv()
        param_state = DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        estimator = _DummyEstimator.any_estimator_with(n_params=1,
                                                       estimation=(np.array([value]), np.random.random()),
                                                       bounds=np.array([[0.1, 0.3]]))

        wrapped = forecasting.ForecastingWrapper(env, param_state, estimator)

        o = wrapped.reset()

        assert o[-1] == pytest.approx(expected, abs=1e-10)

    def test_normalized_params_are_appended_to_observation(self) -> None:
        env = rmenv.DummyEnv()
        param_state = DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        estimator = _DummyEstimator.any_estimator_with(n_params=3,
                                                       estimation=(np.array([0.2, 3., 10.]), np.random.random(size=3)),
                                                       bounds=np.array([[0.1, 0.3],
                                                                        [3., 5.],
                                                                        [5., 20.]]))

        wrapped = forecasting.ForecastingWrapper(env, param_state, estimator)

        o = wrapped.reset()

        assert o[-1] == pytest.approx(np.array([0., -1., -1 / 3]), abs=1e-10)

    def test_true_params_are_exported_in_batch_info(self) -> None:
        param_state = DemandParameterState(arrival_rate=0.1, price_sensitivity=0.25)
        env = continuing.any_continuing_single_leg_env_with(param_state)

        estimator = _DummyEstimator.any_estimator_with()
        wrapped = forecasting.ForecastingWrapper(env, param_state, estimator)

        _ = wrapped.reset()
        _, _, _, info = wrapped.step(wrapped.action_space.sample())

        assert info['orig_obs'][-1] == pytest.approx(.25, abs=1e-10)
        assert info['orig_obs'][-2] == pytest.approx(.1, abs=1e-10)


class TestParamsWrapper(unittest.TestCase):

    def test_params_stats_are_exported_in_batch_info(self) -> None:
        param_state = DemandParameterState(arrival_rate=0.1, price_sensitivity=0.25)
        env = continuing.any_continuing_single_leg_env_with(param_state)

        estimator = _DummyEstimator.any_estimator_with(n_params=2,
                                                       estimation=(np.array([0.15, 0.31]), np.array([0.05, 0.07])),
                                                       param_names=['arrival_rate', 'price_sensitivity'])
        wrapped = forecasting.ParamsWrapper(env, param_state, estimator)

        _ = wrapped.reset()
        _, _, _, info = wrapped.step(wrapped.action_space.sample())

        assert info['estimated_arrival_rate'] == pytest.approx(.15, abs=1e-10)
        assert info['arrival_rate_error'] == pytest.approx((.15 - .1) ** 2, abs=1e-10)
        assert info['arrival_rate_std'] == pytest.approx(0.05, abs=1e-10)

        assert info['estimated_price_sensitivity'] == pytest.approx(.31, abs=1e-10)
        assert info['price_sensitivity_error'] == pytest.approx((.31 - .25) ** 2, abs=1e-10)
        assert info['price_sensitivity_std'] == pytest.approx(0.07, abs=1e-10)


class TestMaximizeRevenueWithStdPenaltyHistoryWrapper:

    @pytest.mark.parametrize(
        'offer_history, expected',
        [(np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]]), np.array([0., 1, 1])),  # oldest record is always deleted

         (np.array([[0, 0, 7],
                    [3, 0, 0],
                    [0, 15, 0]]), np.array([3, 0., 7])),
         ])
    def test_appends_action_distribution_to_observation(self, offer_history: np.ndarray, expected: np.ndarray) -> None:
        _history = continuing.RawHistory(offer_history)
        action_space = spaces.Discrete(offer_history.shape[1])
        env = rmenv.DummyEnv(action_space)
        wrapped = forecasting.MaximizeRevenueWithStdPenaltyHistoryWrapper(env, _history)

        o1 = wrapped.reset()

        assert np.alltrue(o1[-1] == expected)


if __name__ == '__main__':
    unittest.main()

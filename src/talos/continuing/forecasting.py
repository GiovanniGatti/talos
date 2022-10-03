import abc
import math
import random
from typing import Tuple, Any, Dict, List

import gym
import numpy as np
from gym import spaces
from scipy import optimize

from talos import rmenv
from talos.continuing import history


def _estimate_price_sensitivity(offers: np.ndarray,
                                bkgs: np.ndarray,
                                fare_structure: np.ndarray,
                                arrival_rate: float,
                                bound: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    f0 = np.min(fare_structure)

    def neg_log_likelihood(price_sensitivity: float) -> np.ndarray:
        if price_sensitivity < 0:
            raise ValueError(f'price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        return -np.sum(bkgs * np.log(forecast) - offers * forecast)

    def neg_log_likelihood_gradient(price_sensitivity: float) -> np.ndarray:
        if price_sensitivity < 0:
            raise ValueError(f'price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        return -np.sum((fare_structure / f0 - 1) * (offers * forecast - bkgs))

    def log_likelihood_hess(price_sensitivity: float) -> np.ndarray:
        if price_sensitivity < 0:
            raise ValueError(f'price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        return np.sum((fare_structure / f0 - 1) ** 2 * offers * forecast)

    x0 = np.array([random.uniform(min(bound), max(bound))])
    sol = optimize.minimize(
        neg_log_likelihood, x0, method='l-bfgs-b', jac=neg_log_likelihood_gradient, bounds=[bound])

    x = sol.x
    hess = np.clip(log_likelihood_hess(x), a_min=.001, a_max=None)
    hess = np.array([hess])
    std_x = np.sqrt(1 / hess)

    return sol.x, std_x


def _estimate_two_parameters(
        offers: np.ndarray, bkgs: np.ndarray, fare_structure: np.ndarray, bounds: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    f0 = np.min(fare_structure)

    def neg_log_likelihood(params: Tuple[float, float]) -> np.ndarray:
        arrival_rate, price_sensitivity = params
        if price_sensitivity < 0 or arrival_rate < 0:
            raise ValueError(f'arrival_rate = {arrival_rate}, price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        return -np.sum(bkgs * np.log(forecast) - offers * forecast)

    def neg_log_likelihood_gradient(params: Tuple[float, float]) -> np.ndarray:
        arrival_rate, price_sensitivity = params
        if price_sensitivity < 0 or arrival_rate < 0:
            raise ValueError(f'arrival_rate = {arrival_rate}, price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        out = np.empty(2)
        l = bkgs - offers * forecast
        out[0] = -np.sum((1 / arrival_rate) * l)
        out[1] = np.sum((fare_structure / f0 - 1) * l)
        return out

    def log_likelihood_hess(params: Tuple[float, float]) -> np.ndarray:
        arrival_rate, price_sensitivity = params
        if price_sensitivity < 0 or arrival_rate < 0:
            raise ValueError(f'arrival_rate = {arrival_rate}, price_sensitivity = {price_sensitivity}')
        forecast = arrival_rate * np.exp(-price_sensitivity * (fare_structure / f0 - 1))
        out = np.empty((2, 2))
        out[0, 0] = (1 / arrival_rate) * np.sum(bkgs / arrival_rate + offers * forecast)
        out[0, 1] = -(1 / arrival_rate) * np.sum((fare_structure / f0 - 1) * offers * forecast)
        out[1, 0] = out[0, 1]
        out[1, 1] = np.sum((fare_structure / f0 - 1) ** 2 * offers * forecast)
        return out

    _min_arr, _max_arr = min(bounds[0]), max(bounds[0])
    _min_ps, _max_ps = min(bounds[1]), max(bounds[1])
    x0 = np.array([random.uniform(_min_arr, _max_arr), random.uniform(_min_ps, _max_ps)])
    sol = optimize.minimize(neg_log_likelihood, x0, method='l-bfgs-b', jac=neg_log_likelihood_gradient, bounds=bounds)

    x = sol.x
    hess = log_likelihood_hess(x)

    try:
        inv = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return sol.x, np.array([10., 10.])

    inv = np.array([inv[0, 0], inv[1, 1]])
    inv = np.clip(inv, a_min=0., a_max=50.)
    std = np.sqrt(inv)

    return sol.x, std


class EstimatorState(abc.ABC):

    @abc.abstractmethod
    def bounds(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def n_params(self) -> int:
        pass

    @abc.abstractmethod
    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def params(self) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def std(self) -> Dict[str, np.ndarray]:
        pass


class TwoParamsEstimator(EstimatorState):

    def __init__(self,
                 offer_history: history.History,
                 bkg_history: history.History,
                 fare_structure: np.ndarray,
                 bounds: List[Tuple[float, float]]):
        assert len(bounds) == 2, 'Expected bounds for arrival_rate and price_sensitivity'
        self._offer_history = offer_history
        self._bkg_history = bkg_history
        self._fare_structure = fare_structure
        self._bounds = bounds
        self._np_bounds = np.array(self._bounds)
        arrival_rate = np.random.uniform(*bounds[0])
        price_sensitivity = np.random.uniform(*bounds[1])
        self._params, self._std = np.array([arrival_rate, price_sensitivity]), np.array([10., 10.])

    def bounds(self) -> np.ndarray:
        return self._np_bounds

    def n_params(self) -> int:
        return 2

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        offers = self._offer_history.distribution()
        bkgs = self._bkg_history.distribution()
        self._params, self._std = _estimate_two_parameters(offers, bkgs, self._fare_structure, self._bounds)
        return self._params, self._std

    def params(self) -> Dict[str, np.ndarray]:
        return {'arrival_rate': self._params[0], 'price_sensitivity': self._params[1]}

    def std(self) -> Dict[str, np.ndarray]:
        return {'arrival_rate': self._std[0], 'price_sensitivity': self._std[1]}


class PriceSensitivityEstimator(EstimatorState):

    def __init__(self,
                 offer_history: history.History,
                 bkg_history: history.History,
                 fare_structure: np.ndarray,
                 state: rmenv.DemandParameterState,
                 bound: Tuple[float, float]):
        self._offer_history = offer_history
        self._bkg_history = bkg_history
        self._fare_structure = fare_structure
        self._bound = bound
        self._state = state
        self._np_bounds = np.array(self._bound)[np.newaxis, :]
        price_sensitivity = np.random.uniform(*bound)
        self._params, self._std = np.array([price_sensitivity]), np.array([10.])

    def bounds(self) -> np.ndarray:
        return self._np_bounds

    def n_params(self) -> int:
        return 1

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        offers = self._offer_history.distribution()
        bkgs = self._bkg_history.distribution()
        self._params, self._std = _estimate_price_sensitivity(
            offers, bkgs, self._fare_structure, self._state.arrival_rate, self._bound)
        return self._params, self._std

    def params(self) -> Dict[str, np.ndarray]:
        return {'price_sensitivity': self._params}

    def std(self) -> Dict[str, np.ndarray]:
        return {'price_sensitivity': self._std}


class ForecastingWrapper(gym.ObservationWrapper):

    def __init__(self,
                 base_env: gym.Env,
                 state: rmenv.DemandParameterState,
                 estimator_state: EstimatorState):
        super(ForecastingWrapper, self).__init__(base_env)

        self._state = state
        self._estimator_state = estimator_state
        self._max = np.max(estimator_state.bounds(), axis=1)
        self._min = np.min(estimator_state.bounds(), axis=1)

        n_params = estimator_state.n_params()

        space = (base_env.observation_space, spaces.Box(low=-1., high=1., shape=(n_params,), dtype=float))

        self.observation_space = spaces.Tuple(space)

        self._info: Dict[Any, Any] = {}

    def observation(self, observation: Any) -> Tuple[Any, ...]:
        params, std = self._estimator_state.estimate()
        norm_params = 2 * (params - self._min) / (self._max - self._min) - 1
        return observation, norm_params

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        o, r, done, infos = super().step(action)

        if 'orig_obs' in infos:
            infos['orig_obs'] = infos['orig_obs'], self._state.arrival_rate, self._state.price_sensitivity

        assert self._info is not None
        for k, v in self._info.items():
            assert k not in infos
            infos[k] = v

        self._info.clear()

        return o, r, done, infos


class ParamsWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, state: rmenv.DemandParameterState, estimator_state: EstimatorState):
        super().__init__(env)
        self._state = state
        self._estimator_state = estimator_state

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        s, r, done, info = super().step(action)

        acc = 0.
        norm = 0.
        for param_name, value in self._estimator_state.params().items():
            info.update({'estimated_{}'.format(param_name): value})
            if param_name == 'price_sensitivity':
                acc += (self._state.price_sensitivity - value) ** 2
                norm += self._state.price_sensitivity ** 2
                info.update({'price_sensitivity_error': (self._state.price_sensitivity - value) ** 2})
            elif param_name == 'arrival_rate':
                acc += (self._state.arrival_rate - value) ** 2
                norm += self._state.arrival_rate ** 2
                info.update({'arrival_rate_error': (self._state.arrival_rate - value) ** 2})

        if norm > 0:
            acc = 1 - math.sqrt(acc / norm)
            info.update({'accuracy': acc})

        for param_name, value in self._estimator_state.std().items():
            info.update({'{}_std'.format(param_name): value})

        return s, r, done, info


class MaximizeRevenueWithStdPenaltyHistoryWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, offer_history: history.History):
        super(MaximizeRevenueWithStdPenaltyHistoryWrapper, self).__init__(env)
        self._offer_history = offer_history
        shape = (offer_history.nbins(),)
        space = spaces.Box(low=0., high=np.inf, shape=shape, dtype=float)

        if isinstance(env.observation_space, spaces.Tuple):
            def concat_fn(obs: Tuple[Any, ...], distribution: np.ndarray) -> Tuple[Any, ...]:
                return obs + (distribution,)

            self._i = len(env.observation_space)
            self.observation_space = spaces.Tuple(env.observation_space.spaces + (space,))
        else:
            def concat_fn(obs: Any, distribution: np.ndarray) -> Tuple[Any, ...]:
                return obs, distribution

            self._i = 1
            self.observation_space = spaces.Tuple((env.observation_space, space))

        self._concat_fn = concat_fn

    def observation(self, observation: Any) -> Tuple[Any, ...]:
        raw = self._offer_history.raw()
        distribution = np.sum(raw[:-1], axis=0)
        return self._concat_fn(observation, distribution)

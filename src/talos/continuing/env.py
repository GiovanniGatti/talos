import collections
from typing import Any, Dict, Tuple, List

import gym
import numpy as np

from talos import rmenv
from talos.continuing import rwd_fn, policy_state, rwd_wrapper, history, forecasting
from talos.rmenv import sampler


class ContinuingSingleLeg(gym.Env):

    def __init__(self,
                 initial_cap: int,
                 horizon: int,
                 fare_structure: np.ndarray,
                 param_state: rmenv.DemandParameterState,
                 opt_policy: rmenv.PolicyState,
                 _rwd_fn: rwd_fn.RewardFn) -> None:
        self._leg_initial_cap = initial_cap
        self.booking_horizon = horizon
        self._param_state = param_state
        self._fare_structure = fare_structure

        self._f0 = np.min(self._fare_structure).item()

        self._rwd_fn = _rwd_fn
        self._opt_policy = opt_policy

        self.action_space = gym.spaces.MultiDiscrete([len(self._fare_structure)] * self.booking_horizon)
        nvec = np.array([self.booking_horizon, self._leg_initial_cap + 1])
        nvec = np.broadcast_to(nvec, (self.booking_horizon, nvec.shape[0]))
        self.observation_space = gym.spaces.MultiDiscrete(nvec)

        self.random_state = np.random.default_rng()
        self._cap = collections.deque([self._leg_initial_cap] * self.booking_horizon)
        self._dtd = np.arange(1, horizon + 1)
        self._dtd_indexes = self._dtd - 1

    @property
    def initial_capacity(self) -> int:
        return self._leg_initial_cap

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        assert action.shape == (self.booking_horizon,)

        _prev_cap = np.array(self._cap, dtype=int)
        incomplete_flights = _prev_cap > 0
        _idx = self._dtd[incomplete_flights]
        optimal_action = \
            np.sum(self._opt_policy[_idx, _prev_cap[incomplete_flights]] == action[incomplete_flights]) \
            / np.sum(incomplete_flights)

        arrivals = self.random_state.poisson(lam=self._param_state.arrival_rate, size=self.booking_horizon)

        arrivals_dtd_indexes = np.repeat(self._dtd_indexes, arrivals)
        selected_prices = np.repeat(action, arrivals)

        purchase_prob = self._purchase_prob[selected_prices]
        wtp = self.random_state.random(size=np.sum(arrivals)) < purchase_prob
        wtp_dtd_indexes = arrivals_dtd_indexes[wtp]
        wtp_dtd_indexes, count_wtp = np.unique(wtp_dtd_indexes, return_counts=True)
        bookings = np.clip(count_wtp, a_max=[self._cap[i] for i in wtp_dtd_indexes], a_min=0)

        daily_bkgs = np.zeros(self.booking_horizon)
        daily_bkgs[wtp_dtd_indexes] = bookings
        rwd = self._rwd_fn(_prev_cap, daily_bkgs, action)

        for i, b in zip(wtp_dtd_indexes, bookings):
            self._cap[i] -= b
        self._cap.popleft()
        self._cap.append(self._leg_initial_cap)
        _cap = np.array(self._cap, dtype=np.int)

        obs = np.stack((self._dtd_indexes, _cap), axis=-1)

        return obs, rwd, False, {'optimal_action': optimal_action,
                                 'opt_policy': self._opt_policy[self._dtd, _prev_cap],
                                 'raw_rev': np.sum(daily_bkgs * self._fare_structure[action]),
                                 'daily_bkgs': daily_bkgs,
                                 'incomplete_flights': incomplete_flights,
                                 'orig_obs': np.stack((self._dtd_indexes, _prev_cap), axis=-1)}

    def reset(self) -> np.ndarray:
        # noinspection PyAttributeOutsideInit
        # The purchase prob may change after each reset
        self._purchase_prob = np.exp(-self._param_state.price_sensitivity * (self._fare_structure / self._f0 - 1))
        _cap = np.array(self._cap, dtype=np.int)
        return np.stack((self._dtd_indexes, _cap), axis=-1)

    def render(self, mode: str = "human") -> None:
        pass


class WarmupWrapper(gym.Wrapper):

    def __init__(self, _env: gym.Env, warmup_policy: rmenv.PolicyState, horizon: int):
        super().__init__(_env)
        self._warmup_policy = warmup_policy
        self._horizon = horizon

    def reset(self) -> Any:
        obs = super().reset()
        for _ in range(self._horizon):
            _dtd, _cap = obs[:, 0] + 1, obs[:, 1]
            selected_prices = self._warmup_policy[_dtd, _cap]
            obs, _, _, _ = self.step(selected_prices)
        return obs


def with_uniform_sampling(config: Dict[str, Any]) -> gym.Env:
    horizon: int = config['booking-horizon']
    arrivals: List[float] = config['mean-arrivals']
    price_sensitivity: List[float] = config['price-sensitivity']
    assert 0 < len(arrivals) <= 2, 'When using uniform sampling, the user must provide min/max of a ' \
                                   'mean-arrivals range or a single value. mean-arrivals={}'.format(arrivals)
    assert 0 < len(price_sensitivity) <= 2, 'When using uniform sampling, the user must provide min/max of a frat5 ' \
                                            'range or a single value. price_sensitivity={}'.format(price_sensitivity)
    arrival_rate = [arr / horizon for arr in arrivals]
    config['arrival-rate'] = arrival_rate
    return env(config, sampler.UniformPriceSampler(arrival_rate=(min(arrival_rate), max(arrival_rate)),
                                                   price_sensitivity=(min(price_sensitivity), max(price_sensitivity))))


def with_discrete_sampling(config: Dict[str, Any]) -> gym.Env:
    horizon: int = config['booking-horizon']
    arrivals: List[float] = config['mean-arrivals']
    price_sensitivity: List[float] = config['price-sensitivity']
    assert len(arrivals) > 0, 'Expected at least a single value for arrivals. mean-arrivals={}'.format(arrivals)
    assert len(price_sensitivity) > 0, 'Expected at least a single value for price sensitivity. ' \
                                       'price_sensitivity={}'.format(price_sensitivity)
    arrival_rate = [arr / horizon for arr in arrivals]
    config['arrival-rate'] = arrival_rate
    return env(config, sampler.DiscreteSampler(arrival_rate=arrival_rate, price_sensitivity=price_sensitivity))


def env(config: Dict[str, Any], _sampler: rmenv.Sampler) -> gym.Env:
    missing_keys = []
    for k in ['initial-capacity', 'booking-horizon', 'fare-structure', 'rwd-fn', 'one-hot', 'discount-rate',
              'with-true-params', 'warmup-policy']:
        if k not in config.keys():
            missing_keys.append(k)
    if len(missing_keys) > 0:
        raise ValueError('Missing mandatory environment parameters: {}'.format(missing_keys))

    horizon: int = config['booking-horizon']
    fare_structure = np.array(config['fare-structure'], dtype=float)
    initial_capacity: int = config['initial-capacity']
    _rwd_fn: str = config['rwd-fn']
    one_hot: bool = config['one-hot']
    with_true_params: bool = config['with-true-params']
    warmup_policy: str = config['warmup-policy']
    discount_rate: float = config['discount-rate']

    _state = rmenv.DemandParameterState(*_sampler.sample())
    opt_policy_state = policy_state.OptimalPolicy(initial_capacity, horizon, fare_structure)
    rand_policy_state = policy_state.RandomPolicy(initial_capacity, horizon, fare_structure)

    _rwd_fn_instance: rwd_fn.RewardFn
    if _rwd_fn == 'expected-rwd':
        def _rwd_wrapper_fn(e: gym.Env) -> gym.RewardWrapper:
            return rwd_wrapper.NormalizedRewardWrapper(e, opt_policy_state, rand_policy_state)

        _rwd_fn_instance = rwd_fn.ExpectedRwd(initial_capacity, fare_structure, _state)
    elif _rwd_fn == 'stochastic-rwd':
        def _rwd_wrapper_fn(e: gym.Env) -> gym.RewardWrapper:
            return rwd_wrapper.NormalizedRewardWrapper(e, opt_policy_state, rand_policy_state)

        _rwd_fn_instance = rwd_fn.StochasticRwd(fare_structure)
    else:
        raise ValueError('Unexpected reward function {}'.format(_rwd_fn))

    _env = ContinuingSingleLeg(initial_capacity,
                               horizon,
                               fare_structure,
                               _state,
                               opt_policy_state,
                               _rwd_fn_instance)

    if warmup_policy == 'random':
        wrapped = WarmupWrapper(_env, rand_policy_state, horizon)
    elif warmup_policy == 'optimal':
        wrapped = WarmupWrapper(_env, opt_policy_state, horizon)
    elif warmup_policy is None:
        wrapped = _env
    else:
        raise ValueError('Unexpected warmup policy {}'.format(warmup_policy))

    wrapped = _rwd_wrapper_fn(wrapped)
    wrapped = rwd_wrapper.NormalizedReturnWrapper(wrapped, discount_rate)
    wrapped = rmenv.OneHot(wrapped) if one_hot else rmenv.MinMax(wrapped)

    if with_true_params:
        _min_arr, _max_arr = min(config['arrival-rate']), max(config['arrival-rate'])
        _min_ps, _max_ps = min(config['price-sensitivity']), max(config['price-sensitivity'])
        wrapped = rmenv.TrueParamsObsWrapper(wrapped, _state, [(_min_arr, _max_arr), (_min_ps, _max_ps)])

    if 'forecasting' in config:
        forecast: Dict[str, Any] = config['forecasting']
        params_range = forecast['params-range']

        offers_history = history.SumHistory(fare_structure.shape[0], horizon)
        bookings_history = history.SumHistory(fare_structure.shape[0], horizon)
        if len(params_range) > 1:
            estimator_state = forecasting.TwoParamsEstimator(
                offers_history, bookings_history, fare_structure, params_range)
        else:
            estimator_state = forecasting.PriceSensitivityEstimator(
                offers_history, bookings_history, fare_structure, _state, next(iter(params_range)))
        wrapped = forecasting.ForecastingWrapper(wrapped, _state, estimator_state)
        wrapped = forecasting.ParamsWrapper(wrapped, _state, estimator_state)
        wrapped = history.OfferHistoryWrapper(wrapped, offers_history)
        wrapped = history.BookingHistoryWrapper(wrapped, bookings_history)

        if 'eta' in config:
            wrapped = forecasting.MaximizeRevenueWithStdPenaltyHistoryWrapper(wrapped, offers_history)

    wrapped = rmenv.SampleParameterWrapper(wrapped, _state, _sampler, [opt_policy_state, rand_policy_state])

    return wrapped

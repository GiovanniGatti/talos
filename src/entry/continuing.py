import argparse
import math
import multiprocessing
import pathlib
import sys
import tempfile
from typing import Optional, Any

import ray
from ray import tune, rllib
from ray.rllib.agents import callbacks, ppo
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import Dict
from ray.tune.analysis import experiment_analysis

from talos.continuing import env, model, postprocess

_, tf, _ = try_import_tf()


class TrackingCallback(callbacks.DefaultCallbacks):

    def on_episode_step(self, *,
                        worker: rllib.RolloutWorker,
                        base_env: rllib.BaseEnv,
                        episode: rllib.evaluation.episode.MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs: Dict[Any, Any]) -> None:
        super().on_episode_step(worker=worker, base_env=base_env, episode=episode, env_index=env_index, **kwargs)
        info = episode.last_info_for()
        if info:
            for k in {'optimal_action', 'raw_rev', 'rwd_before_norm', 'accuracy', 'price_sensitivity_std',
                      'price_sensitivity_error', 'arrival_rate_std', 'arrival_rate_error'}:
                if k in info:
                    data = episode.user_data.setdefault(k, [])
                    data.append(info[k])

    def on_episode_end(self, *,
                       worker: rllib.RolloutWorker,
                       base_env: rllib.BaseEnv,
                       policies: Dict[rllib.utils.typing.PolicyID, rllib.Policy],
                       episode: rllib.evaluation.episode.MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs: Dict[Any, Any]) -> None:
        super().on_episode_end(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)
        horizon = next(iter(base_env.get_unwrapped())).unwrapped.booking_horizon
        for k, v in episode.user_data.items():
            episode.custom_metrics[k] = v[horizon:]


def recover_from_preemption(local_dir: str) -> Optional[str]:
    _local_dir = pathlib.Path(local_dir)
    local_checkpoints = sorted(_local_dir.glob('**/checkpoint_*[0-9]'), reverse=True)
    if len(local_checkpoints) > 0:
        checkpoint_dir = pathlib.Path(next(iter(local_checkpoints)))
        checkpoint_file = next(checkpoint_dir.glob('checkpoint-*[0-9]'))
        return str(checkpoint_file)
    return None


def main(_namespace: argparse.Namespace, _tmp_dir: str) -> experiment_analysis.ExperimentAnalysis:
    booking_horizon: int = _namespace.horizon

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    num_cpus = multiprocessing.cpu_count()

    if _namespace.debugging:
        num_workers = 1
        num_envs_per_worker = 32
        years = 3
        sgd_minibatch_size = 128
        num_sgd_iter = 2
        framework = 'tf2'
        local_dir = str(pathlib.Path(_tmp_dir, 'ray-results'))
    else:
        num_workers = num_cpus - 1
        years = 20
        num_envs_per_worker = int(math.ceil(_namespace.batch_size / (num_workers * years * booking_horizon)))
        sgd_minibatch_size = _namespace.minibatch_size
        num_sgd_iter = _namespace.num_sgd_iter
        framework = 'tf'
        local_dir = './logs/ray-results'

    train_batch_size = num_workers * num_envs_per_worker * years * booking_horizon

    _checkpoint = recover_from_preemption(local_dir)
    if _checkpoint is not None:
        print('Using checkpoint file \'{}\' before preemption'.format(_checkpoint))

    ModelCatalog.register_custom_model('time-distributed', model.TimeDistributedModel)
    ModelCatalog.register_custom_model('encoder-decoder', model.EncoderDecoderModel)
    tune.register_env('with-uniform-sampling', env.with_uniform_sampling)

    _validate_config_fn, _postprocess_fn = postprocess.no_postprocess()
    gamma = _namespace.gamma
    vf_clip_param = 30.
    if _namespace.post_process == 'avg-cost':
        _validate_config_fn, _postprocess_fn = postprocess.avg_cost()
        gamma = 1.
    elif _namespace.post_process == 'rvi':
        _validate_config_fn, _postprocess_fn = postprocess.rvi()
        gamma = 1.
        vf_clip_param = 1.
    elif _namespace.post_process == 'diff-td':
        _validate_config_fn, _postprocess_fn = postprocess.diff_td()
        gamma = 1.

    CustomPPOPolicy = ppo.PPOTFPolicy.with_updates(postprocess_fn=_postprocess_fn)
    CustomPPOTrainer = ppo.PPOTrainer.with_updates(validate_config=_validate_config_fn, default_policy=CustomPPOPolicy)

    env_config = {
        'booking-horizon': namespace.horizon,
        'initial-capacity': namespace.initial_capacity,
        'mean-arrivals': namespace.mean_arrivals,
        'rwd-fn': namespace.rwd_fn,
        'price-sensitivity': [math.log(2) / (frat5 - 1.) for frat5 in namespace.frat5],
        'one-hot': namespace.one_hot,
        'discount-rate': gamma,
        'fare-structure': tuple(range(50, 250, 20)),
        'with-true-params': not namespace.with_forecasting and (
                (namespace.frat5 and len(namespace.frat5) > 1) or
                (namespace.mean_arrivals and len(namespace.mean_arrivals) > 1) or
                namespace.env == 'with-uniform-sampling'),
        'warmup-policy': namespace.warmup_policy,
    }

    if namespace.with_forecasting:
        assert len(_namespace.frat5) > 1, 'Expected 2+ frat5s in forecasting mode'

        params_range = []
        if len(_namespace.mean_arrivals) > 1:
            arrival_rate_range = min(_namespace.mean_arrivals) / booking_horizon, \
                                 max(_namespace.mean_arrivals) / booking_horizon
            params_range += [arrival_rate_range, ]

        price_sensitivity_range = math.log(2) / (max(_namespace.frat5) - 1), \
                                  math.log(2) / (min(_namespace.frat5) - 1)
        params_range += [price_sensitivity_range, ]

        env_config['forecasting'] = {
            'params-range': params_range,
        }

    return tune.run(
        run_or_experiment=CustomPPOTrainer,
        config={
            # Training settings
            'env': 'with-uniform-sampling',
            'env_config': env_config,
            'num_workers': num_workers,
            'num_cpus_per_worker': 1,
            'num_envs_per_worker': num_envs_per_worker,
            'rollout_fragment_length': years * booking_horizon,
            'framework': framework,

            'model': {
                'custom_model': namespace.model,
                'fcnet_activation': 'relu',
                'lstm_cell_size': namespace.lstm_cell_size,
            },

            'lr': _namespace.lr,
            "entropy_coeff": _namespace.entropy_coeff,
            'vf_loss_coeff': namespace.vf_loss_coeff,

            # Continuing Task settings
            'gamma': gamma,
            'horizon': years * booking_horizon,
            'lambda': vars(_namespace)['lambda'],
            'soft_horizon': False,
            'no_done_at_end': True,

            'num_cpus_for_driver': 1,
            'tf_session_args': {
                'intra_op_parallelism_threads': 0,
                'inter_op_parallelism_threads': 0,
                'log_device_placement': False,
                'device_count': {'CPU': 1},
                'gpu_options': {'allow_growth': True},
                'allow_soft_placement': True,
            },
            'local_tf_session_args': {
                'intra_op_parallelism_threads': 0,
                'inter_op_parallelism_threads': 0,
            },

            'num_gpus': num_gpus,

            # PPO specific
            'train_batch_size': train_batch_size,
            'sgd_minibatch_size': sgd_minibatch_size,
            'num_sgd_iter': num_sgd_iter,
            'vf_clip_param': vf_clip_param,

            # Policy evaluation config
            'evaluation_interval': 0,

            'callbacks': TrackingCallback,
        },
        restore=_checkpoint,
        checkpoint_freq=5,
        stop=lambda t, r: (r['info']['learner']['default_policy']['learner_stats']['entropy'] <= _namespace.stop or
                           r['training_iteration'] >= _namespace.max_iterations),
        checkpoint_at_end=True,
        raise_on_failed_trial=False,
        local_dir=local_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyper-parameters
    parser.add_argument('--minibatch-size', type=int, default=4096, help='The sgd minibatch size')
    parser.add_argument('--batch-size', type=int, default=1_768_000, help='The sgd minibatch size')
    parser.add_argument('--num-sgd-iter', type=int, default=30, help='The number of sgd iterations per training step')
    parser.add_argument('--entropy-coeff', type=float, default=1.5e-2,
                        help='The weight to the entropy coefficient in the loss function')
    parser.add_argument('--gamma', type=float, default=.95, help='The discount rate')
    parser.add_argument('--lambda', type=float, default=.1, help='The eligibility trace')
    parser.add_argument('--vf-loss-coeff', type=float, default=1.,
                        help='The value loss coefficient (optimize it if actor and critic share layers)')
    parser.add_argument('--lr', type=float, default=3e-05, help='The learning rate')

    # artificial neural network settings
    parser.add_argument('--model', type=str, default='encoder-decoder',
                        choices=['time-distributed', 'encoder-decoder'],
                        help='The ANN architecture to train')
    parser.add_argument('--lstm-cell-size', type=int, default=128,
                        help='The lstm model cell size to use (compatible only with encoder-decoder architecture)')

    # environment settings
    parser.add_argument('--initial-capacity', type=int, default=50, help='The leg capacity')
    parser.add_argument('--horizon', type=int, default=22, help='The booking horizon')
    parser.add_argument('--mean-arrivals', type=float, default=[50., 85.], nargs='+',
                        help='If using the discrete environment, it\'s a list of mean-arrival to train the agent with. '
                             'If using the uniform environment, it\'s the min/max range for mean arrivals')
    parser.add_argument('--frat5', type=float, nargs='+', default=[1.5, 4.3],
                        help='If using the discrete environment, it\'s a list of frat5s to train the agent with. '
                             'If using the uniform environment, it\'s the min/max range for the frat5')
    parser.add_argument('--rwd-fn', type=str, default='expected-rwd',
                        choices=['expected-rwd', 'stochastic-rwd', ],
                        help='The rewarding function to use. The expected-rwd computes, at each time step, the '
                             'expectation of the obtained revenue. This signal has been found to be very reliable '
                             'and it speeds learning. Or, one can use the more classical stochastic-rwd approach, that '
                             'returns the number of bookings times the fare.')

    # miscellaneous
    parser.add_argument('--post-process', type=str, choices=['rvi', 'avg-cost', 'diff-td'],
                        help='There are many ways to compute the average reward. In the paper, we use the differential '
                             'TD approach, but many others can be used, see https://arxiv.org/pdf/2006.16318.pdf')
    parser.add_argument('--one-hot', action='store_true', help='Activate one-hot encoding for agent observations. '
                                                               'We recommend the default min-max normalization.')
    parser.add_argument('--stop', type=float, default=1.5, help='The policy entropy value which training stops')
    parser.add_argument('--max-iterations', type=int, default=205, help='The maximum number of training iterations')
    parser.add_argument('--warmup-policy', type=str, choices=['optimal', 'random'],
                        help='Policy to be used when initializing the environment. If unspecified, '
                             'the environment will use a soft-initialization, i.e., there will be no refresh on '
                             'historical data, each new training episode continues from the previous one.')

    # forecasting
    parser.add_argument('--with-forecasting', action='store_true',
                        help='Use price sensitivity estimation module from RMS')

    # debugging
    parser.add_argument('--debugging', action='store_true', help='Run locally with simplified settings')

    namespace = parser.parse_args(sys.argv[1:])

    ray.init(local_mode=namespace.debugging)

    tmp_dir = tempfile.mkdtemp()
    analysis = main(namespace, tmp_dir)
    while True:
        # Re-launch trial when 'Assert agent_key not in self.agent_collectors' is bug fixed
        # We can safely remove it when https://github.com/ray-project/ray/issues/15297 is closed.
        relaunch = False
        incomplete_trials = []
        for trial in analysis.trials:
            if trial.status == experiment_analysis.Trial.ERROR:
                if 'assert agent_key not in self.agent_collectors' in trial.error_msg:
                    relaunch = True
                else:
                    incomplete_trials.append(trial)

        if incomplete_trials:
            raise tune.TuneError("Trials did not complete", incomplete_trials)

        if relaunch:
            analysis = main(namespace, tmp_dir)
            continue
        break

    sys.exit(0)

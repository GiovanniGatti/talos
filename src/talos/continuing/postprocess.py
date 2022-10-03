from typing import Optional, Dict, Tuple, Callable

import numpy as np
from ray import rllib
from ray.rllib import evaluation
from ray.rllib.agents.ppo import ppo
from ray.rllib.evaluation import postprocessing
from ray.rllib.policy import sample_batch
from ray.rllib.utils import typing

ValidationConfigFn = Callable[[typing.TrainerConfigDict], None]

PostProcessFn = Callable[[rllib.Policy, sample_batch.SampleBatch,
                          Optional[Dict[typing.AgentID, sample_batch.SampleBatch]],
                          Optional[evaluation.MultiAgentEpisode]],
                         sample_batch.SampleBatch]


def _validate_gamma_is_one(config: typing.TrainerConfigDict) -> None:
    ppo.validate_config(config)
    if not config['gamma'] == 1.:
        raise ValueError('\'gamma: {}\' should be 1. when using differential return settings'.format(config['gamma']))


def _with_avg_rwd_fn(
        avg_rwd_fn: Callable[[sample_batch.SampleBatch, np.ndarray], None]
) -> PostProcessFn:
    def _avg_rwd_advantages(
            policy: rllib.Policy,
            rollout: sample_batch.SampleBatch,
            other_agent_batches: Optional[Dict[typing.AgentID, sample_batch.SampleBatch]] = None,
            episode: Optional[evaluation.MultiAgentEpisode] = None
    ) -> sample_batch.SampleBatch:
        # Heavily copied and pasted from RLlib (ray.rllib.evalutation.postprocessing) with minor adaptations
        assert not np.all(rollout[sample_batch.SampleBatch.DONES])
        # Create an input dict according to the Model's requirements.
        # noinspection PyTypeChecker
        # following RLlib implementation
        input_dict = rollout.get_single_step_input_dict(policy.model.view_requirements, index='last')
        # noinspection PyProtectedMember
        # following RLlib implementation
        last_r = policy._value(**input_dict)
        vpred_t = np.concatenate([rollout[sample_batch.SampleBatch.VF_PREDS], np.array([last_r])])

        avg_rwd_fn(rollout, vpred_t)

        batch = postprocessing.compute_advantages(
            rollout,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True))

        return batch

    return _avg_rwd_advantages


def avg_cost_fn(rollout: sample_batch.SampleBatch, vpred_t: np.ndarray) -> None:
    mean_rwd = np.mean(rollout[sample_batch.SampleBatch.REWARDS])
    rollout[sample_batch.SampleBatch.REWARDS] -= mean_rwd


def diff_td_fn(rollout: sample_batch.SampleBatch, vpred_t: np.ndarray) -> None:
    mean_delta_t = np.mean(rollout[sample_batch.SampleBatch.REWARDS] + vpred_t[1:] - vpred_t[:-1])
    rollout[sample_batch.SampleBatch.REWARDS] -= mean_delta_t


def rvi_fn(rollout: sample_batch.SampleBatch, vpred_t: np.ndarray) -> None:
    rollout[sample_batch.SampleBatch.REWARDS] -= vpred_t[0]


def no_postprocess() -> Tuple[ValidationConfigFn, PostProcessFn]:
    return ppo.validate_config, postprocessing.compute_gae_for_sample_batch


def avg_cost() -> Tuple[ValidationConfigFn, PostProcessFn]:
    return _validate_gamma_is_one, _with_avg_rwd_fn(avg_cost_fn)


def diff_td() -> Tuple[ValidationConfigFn, PostProcessFn]:
    return _validate_gamma_is_one, _with_avg_rwd_fn(diff_td_fn)


def rvi() -> Tuple[ValidationConfigFn, PostProcessFn]:
    return _validate_gamma_is_one, _with_avg_rwd_fn(rvi_fn)

from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import spaces
from ray.rllib.models.tf import tf_modelv2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType

_, tf, _ = try_import_tf()


class TimeDistributedModel(tf_modelv2.TFModelV2):

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):

        self.original_space = obs_space.original_space if hasattr(obs_space, 'original_space') else obs_space

        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise ValueError('Expected action space of type MultiDiscrete, but found {}' % action_space)

        if np.unique(action_space.nvec).shape != (1,):
            raise ValueError(
                'Expected action space to have a uniform vector probability size, '
                'but it has multiple shape sizes: {}'.format(np.unique(action_space.nvec)))

        super(TimeDistributedModel, self).__init__(self.original_space, action_space, num_outputs, model_config, name)

        self.horizon = next(iter(action_space.nvec.shape))
        self.n_fares = np.unique(action_space.nvec).item()

        model_inputs = []
        actor_inputs = []
        critic_inputs = []
        if isinstance(self.original_space, gym.spaces.Tuple):
            # using price sensitivity / action distribution / noise std
            state_obs_input = self.original_space[0].shape[1]
            state_inputs = tf.keras.layers.Input(shape=self.original_space[0].shape, name='state_obs')

            for extra in self.original_space[1:]:
                raw_input = tf.keras.layers.Input(shape=extra.shape)
                repeated_input = tf.keras.layers.RepeatVector(self.horizon)(raw_input)
                model_inputs += [raw_input, ]
                actor_inputs += [repeated_input, ]
                critic_inputs += [raw_input, ]
        elif isinstance(self.original_space, gym.spaces.Box):
            state_obs_input = obs_space.shape[1]
            state_inputs = tf.keras.layers.Input(shape=obs_space.shape, name='state_obs')
        else:
            raise ValueError('Unsupported observation space {}'.format(self.original_space))

        if state_obs_input > 2:
            # managing one-hot encoding
            flattened_states = state_inputs[:, :, self.horizon:]
            flattened_states = tf.keras.layers.Flatten()(flattened_states)
        else:
            flattened_states = state_inputs[:, :, -1]

        model_inputs[:0] += [state_inputs, ]
        actor_inputs[:0] = [state_inputs, ]
        critic_inputs[:0] = [flattened_states, ]

        fcnet_activation = model_config['fcnet_activation']

        actor_tensor = tf.keras.layers.Concatenate()(actor_inputs) if len(actor_inputs) > 1 else actor_inputs[0]
        policy_layer = tf.keras.layers.Dense(128, activation=fcnet_activation, name='policy_hidden_0')(actor_tensor)
        policy_layer = tf.keras.layers.Dense(128, activation=fcnet_activation, name='policy_hidden_1')(policy_layer)
        policy_layer = tf.keras.layers.Dense(self.n_fares, activation=None, name='policy_out')(policy_layer)

        critic_tensor = tf.keras.layers.Concatenate()(critic_inputs) if len(critic_inputs) > 1 else critic_inputs[0]
        vf_layer = tf.keras.layers.Dense(128, activation=fcnet_activation, name='value_hidden_0')(critic_tensor)
        vf_layer = tf.keras.layers.Dense(128, activation=fcnet_activation, name='value_hidden_1')(vf_layer)
        vf_out = tf.keras.layers.Dense(1, activation=None, name='value_out')(vf_layer)

        self.base_model = tf.keras.Model(model_inputs, [policy_layer, vf_out])

    def forward(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        # noinspection PyAttributeOutsideInit
        # following pattern used by RLLib
        model_out, self._value_out = self.base_model(input_dict['obs'])
        model_out = tf.reshape(model_out, [-1, self.n_fares * self.horizon])
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, h5_file: str) -> None:
        self.base_model.load_weights(h5_file)


class EncoderDecoderModel(tf_modelv2.TFModelV2):

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):

        self.original_space = obs_space.original_space if hasattr(obs_space, 'original_space') else obs_space

        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise ValueError('Expected action space of type MultiDiscrete, but found {}' % action_space)

        if np.unique(action_space.nvec).shape != (1,):
            raise ValueError(
                'Expected action space to have a uniform vector probability size, '
                'but it has multiple shape sizes: {}'.format(np.unique(action_space.nvec)))

        super(EncoderDecoderModel, self).__init__(self.original_space, action_space, num_outputs, model_config, name)

        self.horizon = next(iter(action_space.nvec.shape))
        self.n_fares = np.unique(action_space.nvec).item()

        lstm_size = model_config['lstm_cell_size']

        time_inputs = tf.keras.layers.Input(shape=(self.horizon, self.horizon), name='time_inputs')
        model_inputs = [time_inputs, ]
        actor_inputs = [time_inputs, ]

        state_inputs = tf.keras.layers.Input(shape=self.horizon, name='state_obs')
        model_inputs += [state_inputs, ]

        enc_inputs = [tf.expand_dims(state_inputs, -1), ]
        if isinstance(self.original_space, spaces.Tuple):
            params_input = tf.keras.layers.Input(shape=self.original_space[1].shape, name='params')
            params = tf.keras.layers.RepeatVector(self.horizon)(params_input)
            model_inputs += [params_input, ]
            enc_inputs += [params, ]

            if len(self.original_space) > 2:
                shift = '-{}:0'.format(self.horizon - 1)
                self.view_requirements.update({
                    'prev_obs': ViewRequirement(data_col='obs', shift=shift, space=obs_space),
                })
                self._history_shape = (self.horizon, sum(space.shape[0] for space in self.original_space[2:]))
                history = tf.keras.layers.Input(shape=self._history_shape, name='history')
                model_inputs += [history, ]
                enc_inputs += [history, ]

        concat = tf.keras.layers.Concatenate()(enc_inputs) if len(enc_inputs) > 1 else enc_inputs[0]
        actor_enc, actor_h, actor_c = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True, name='actor_enc')(concat)
        _, critic_h, critic_c = tf.keras.layers.LSTM(
            lstm_size, return_sequences=False, return_state=True, name='critic_enc')(concat)
        critic_tensor = tf.keras.layers.Concatenate()([critic_h, critic_c])
        critic_inputs = [critic_tensor, ]

        actor_tensor = tf.keras.layers.Concatenate()(actor_inputs) if len(actor_inputs) > 1 else actor_inputs[0]

        policy_layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True, name='actor_dec')(
            actor_tensor, initial_state=[actor_h, actor_c])
        attn_out = tf.keras.layers.Attention(name='actor_attn')([actor_enc, policy_layer])
        policy_layer = tf.keras.layers.Concatenate(axis=-1, )([policy_layer, attn_out])
        policy_layer = tf.keras.layers.Dense(self.n_fares, activation=None, name='policy_out')(policy_layer)

        critic_tensor = tf.keras.layers.Concatenate()(critic_inputs) if len(critic_inputs) > 1 else critic_inputs[0]
        vf_out = tf.keras.layers.Dense(1, activation=None, name='value_out')(critic_tensor)

        self.base_model = tf.keras.Model(model_inputs, [policy_layer, vf_out])
        pass

    def forward(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        orig_obs = input_dict['obs']
        orig_state_obs = orig_obs[0] if isinstance(self.original_space, spaces.Tuple) else orig_obs
        batch_size = tf.shape(orig_state_obs)[0]
        dtds = tf.tile(tf.one_hot(tf.range(self.horizon), self.horizon), [batch_size, 1])
        dtds = tf.reshape(dtds, (batch_size, self.horizon, self.horizon))
        states = orig_state_obs[:, :, 1]
        obs = [dtds, states]
        if isinstance(self.original_space, spaces.Tuple):
            obs += [orig_obs[1], ]
            if 'prev_obs' in input_dict:
                history = input_dict['prev_obs'][:, :, -self._history_shape[1]:]
                obs += [history, ]
        # noinspection PyAttributeOutsideInit
        # following pattern used by RLLib
        model_out, self._value_out = self.base_model(obs)
        model_out = tf.reshape(model_out, [-1, self.n_fares * self.horizon])
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, h5_file: str) -> None:
        self.base_model.load_weights(h5_file)

import gym
import numpy as np
import tensorflow as tf
from flatland.core.grid import grid4
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from models.common.models import NatureCNN, ImpalaCNN


class GlobalObsModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."
        self._action_space = action_space
        self._options = model_config['custom_options']
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)

        if self._mask_unavailable_actions:
            obs_space = obs_space['obs']
        else:
            obs_space = obs_space

        observations = tf.keras.layers.Input(shape=obs_space.shape)
        processed_observations = observations # preprocess_obs(tuple(observations))

        if self._options['architecture'] == 'nature':
            conv_out = NatureCNN(activation_out=True, **self._options['architecture_options'])(processed_observations)
        elif self._options['architecture'] == 'impala':
            conv_out = ImpalaCNN(activation_out=True, **self._options['architecture_options'])(processed_observations)
        else:
            raise ValueError(f"Invalid architecture: {self._options['architecture']}.")
        logits = tf.keras.layers.Dense(units=action_space.n)(conv_out)
        baseline = tf.keras.layers.Dense(units=1)(conv_out)
        self._model = tf.keras.Model(inputs=observations, outputs=[logits, baseline])
        self.register_variables(self._model.variables)
        # self._model.summary()

    def forward(self, input_dict, state, seq_lens):
        # obs = preprocess_obs(input_dict['obs'])
        if self._mask_unavailable_actions:
            obs = input_dict['obs']['obs']
        else:
            obs = input_dict['obs']
        logits, baseline = self._model(obs)
        self.baseline = tf.reshape(baseline, [-1])
        if self._mask_unavailable_actions:
            inf_mask = tf.maximum(tf.math.log(input_dict['obs']['available_actions']), tf.float32.min)
            logits = logits + inf_mask
        return logits, state

    def variables(self):
        return self._model.variables

    def value_function(self):
        return self.baseline


def preprocess_obs(obs) -> tf.Tensor:
    transition_map, agents_state, targets = obs

    processed_agents_state_layers = []
    for i, feature_layer in enumerate(tf.unstack(agents_state, axis=-1)):
        if i in {0, 1}:  # agent direction (categorical)
            feature_layer = tf.one_hot(tf.cast(feature_layer, tf.int32), depth=len(grid4.Grid4TransitionsEnum) + 1,
                                       dtype=tf.float32)
        elif i in {2, 4}:  # counts
            feature_layer = tf.expand_dims(tf.math.log(feature_layer + 1), axis=-1)
        else:  # well behaved scalars
            feature_layer = tf.expand_dims(feature_layer, axis=-1)
        processed_agents_state_layers.append(feature_layer)

    return tf.concat(
        [tf.cast(transition_map, tf.float32), tf.cast(targets, tf.float32)] + processed_agents_state_layers, axis=-1)

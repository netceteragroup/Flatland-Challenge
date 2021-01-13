import gym
import numpy as np
import tensorflow as tf
from flatland.core.grid import grid4
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from models.common.models import NatureCNN, ImpalaCNN


class GlobalDensObsModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."
        self._action_space = action_space
        self._options = model_config['custom_options']
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)

        if self._mask_unavailable_actions:
            obs_space = obs_space.original_space['obs']
        else:
            obs_space = obs_space.original_space

        observations = [tf.keras.layers.Input(shape=o.shape) for o in obs_space]
        comp_obs = tf.concat(observations, axis=-1)

        if self._options['architecture'] == 'nature':
            conv_out = NatureCNN(activation_out=True, **self._options['architecture_options'])(comp_obs)
        elif self._options['architecture'] == 'impala':
            conv_out = ImpalaCNN(activation_out=True, **self._options['architecture_options'])(comp_obs)
        else:
            raise ValueError(f"Invalid architecture: {self._options['architecture']}.")
        logits = tf.keras.layers.Dense(units=action_space.n)(conv_out)
        baseline = tf.keras.layers.Dense(units=1)(conv_out)
        self._model = tf.keras.Model(inputs=observations, outputs=[logits, baseline])
        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
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

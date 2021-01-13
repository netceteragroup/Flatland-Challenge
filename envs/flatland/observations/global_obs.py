import gym
import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.core.grid import grid4
from envs.flatland.observations import Observation, register_obs

'''
A 2-d array matrix on-hot encoded similar to tf.one_hot function
https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
'''
def one_hot2d(arr,depth):
    return (np.arange(depth) == arr[...,None]).astype(int)

def preprocess_obs(obs):
    transition_map, agents_state, targets = obs
    new_agents_state = agents_state.transpose([2,0,1])
    *states, = new_agents_state
    processed_agents_state_layers = []
    for i, feature_layer in enumerate(states):
        if i in {0, 1}:  # agent direction (categorical)
            # feature_layer = tf.one_hot(tf.cast(feature_layer, tf.int32), depth=len(grid4.Grid4TransitionsEnum) + 1,
            #                            dtype=tf.float32).numpy()
            # Numpy Version
            feature_layer = one_hot2d(feature_layer, depth=len(grid4.Grid4TransitionsEnum) + 1)
        elif i in {2, 4}:  # counts
            feature_layer = np.expand_dims(np.log(feature_layer + 1), axis=-1)
        else:  # well behaved scalars
            feature_layer = np.expand_dims(feature_layer, axis=-1)
        processed_agents_state_layers.append(feature_layer)

    return np.concatenate([transition_map, targets] + processed_agents_state_layers, axis=-1)

@register_obs("global")
class GlobalObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PaddedGlobalObsForRailEnv(max_width=config['max_width'], max_height=config['max_height'])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        grid_shape = (self._config['max_width'], self._config['max_height'])
        return gym.spaces.Box(low=0, high=np.inf, shape=grid_shape + (31,), dtype=np.float32)


class PaddedGlobalObsForRailEnv(ObservationBuilder):

    def __init__(self, max_width, max_height):
        super().__init__()
        self._max_width = max_width
        self._max_height = max_height
        self._builder = GlobalObsForRailEnv()

    def set_env(self, env: Environment):
        self._builder.set_env(env)

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs = list(self._builder.get(handle))
        height, width = obs[0].shape[:2]
        pad_height, pad_width = self._max_height - height, self._max_width - width
        obs[1] = obs[1] + 1  # get rid of -1
        assert pad_height >= 0 and pad_width >= 0
        return preprocess_obs(tuple([
            np.pad(o, ((0, pad_height), (0, pad_height), (0, 0)), constant_values=0)
            for o in obs
        ]))

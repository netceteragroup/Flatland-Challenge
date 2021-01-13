import gym
import numpy as np
from typing import Optional, List, Dict
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from envs.flatland.observations import Observation, register_obs


@register_obs("density")
class ProjectedDensityObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = ProjectedDensityForRailEnv(config['height'], config['width'], config['encoding'], config['max_t'])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        obs_shape = self._builder.observation_shape
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),
            gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),
        ])


class ProjectedDensityForRailEnv(ObservationBuilder):

    def __init__(self, height, width, encoding='exp_decay', max_t=10):
        super().__init__()
        self._height = height
        self._width = width
        self._depth = max_t + 1 if encoding == 'series' else 1
        if encoding == 'exp_decay':
            self._encode = lambda t: np.exp(-t / np.sqrt(max_t))
        elif encoding == 'lin_decay':
            self._encode = lambda t: (max_t - t) / max_t
        else:
            self._encode = lambda t: 1
        self._predictor = ShortestPathPredictorForRailEnv(max_t)

    @property
    def observation_shape(self):
        return (self._height, self._width, self._depth)

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        get density maps for agents and compose the observation with agent's and other's density maps
        """
        self._predictions = self._predictor.get()
        density_maps = dict()
        for handle in handles:
            density_maps[handle] = self.get(handle)
        obs = dict()
        for handle in handles:
            other_dens_maps = [density_maps[key] for key in density_maps if key != handle]
            others_density = np.mean(np.array(other_dens_maps), axis=0)
            obs[handle] = [density_maps[handle], others_density]
        return obs

    def get(self, handle: int = 0):
        """
        compute density map for agent: a value is asigned to every cell along the shortest path between
        the agent and its target based on the distance to the agent, i.e. the number of time steps the
        agent needs to reach the cell, encoding the time information.
        """
        density_map = np.zeros(shape=(self._height, self._width, self._depth), dtype=np.float32)
        agent = self.env.agents[handle]
        if self._predictions[handle] is not None:
            for t, prediction in enumerate(self._predictions[handle]):
                p = tuple(np.array(prediction[1:3]).astype(int))
                d = t if self._depth > 1 else 0
                density_map[p][d] = self._encode(t)
        return density_map

    def set_env(self, env: Environment):
        self.env: RailEnv = env
        self._predictor.set_env(self.env)

    def reset(self):
        pass

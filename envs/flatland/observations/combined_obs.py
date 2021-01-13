import gym
from flatland.core.env_observation_builder import ObservationBuilder
from typing import Optional, List

from envs.flatland.observations import Observation, register_obs, make_obs


@register_obs("combined")
class CombinedObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._observations = [
            make_obs(obs_name, config[obs_name]) for obs_name in config.keys()
        ]
        self._builder = CombinedObsForRailEnv([
            o._builder for o in self._observations
        ])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        space = []
        for o in self._observations:
            space.append(o.observation_space())
        return gym.spaces.Tuple(space)


class CombinedObsForRailEnv(ObservationBuilder):

    def __init__(self, builders: [ObservationBuilder]):
        super().__init__()
        self._builders = builders

    def reset(self):
        for b in self._builders:
            b.reset()

    def get(self, handle: int = 0):
        return None

    def get_many(self, handles: Optional[List[int]] = None):
        obs = {h: [] for h in handles}
        for b in self._builders:
            sub_obs = b.get_many(handles)
            for h in handles:
                obs[h].append(sub_obs[h])
        return obs

    def set_env(self, env):
        for b in self._builders:
            b.set_env(env)

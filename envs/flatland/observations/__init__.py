import importlib
import os
from abc import ABC, abstractmethod

import gym
import humps
from flatland.core.env_observation_builder import ObservationBuilder


class Observation(ABC):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def builder(self) -> ObservationBuilder:
        pass

    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass


OBS_REGISTRY = {}


def register_obs(name: str):
    def register_observation_cls(cls):
        if name in OBS_REGISTRY:
            raise ValueError(f'Observation "{name}" already registred.')
        if not issubclass(cls, Observation):
            raise ValueError(f'Observation "{name}" ({cls.__name__}) must extend the Observation base class.')
        OBS_REGISTRY[name] = cls
        return cls

    return register_observation_cls


def make_obs(name: str, config, *args, **kwargs) -> Observation:
    return OBS_REGISTRY[name](config, *args, **kwargs)


# automatically import any Python files in the obs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        basename = os.path.basename(file)
        filename = basename.replace(".py", "")
        class_name = humps.pascalize(filename)

        module = importlib.import_module(f'.{file[:-3]}', package=__name__)
        #print("-    Successfully Loaded Observation class {} from {}".format(
        #    class_name, basename
        #))

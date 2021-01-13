from flatland.envs.agent_utils import RailAgentStatus
from flatland.evaluators.client import FlatlandRemoteClient
import numpy as np
import time
import yaml
import ray
import gym
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents import dqn
from envs.flatland.observations.custom_graph_obs import CustomGraphObservation, Features
from utils.loader import load_models
import os
from envs.flatland.utils.gym_env_wrappers import available_actions
load_models(os.getcwd())

class FlatlandMinimalWrapper(gym.Env):
    def __init__(self, env):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'obs':  gym.spaces.Box(low=-1, high=np.inf, shape=(len(Features._fields),)),
            'available_actions': gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)
        })
        self._env = None

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

with open("apex_s_bigger_obs.yaml") as f:
    experiments = yaml.safe_load(f)

ray.init(configure_logging=False)
config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
experiment_name = list(experiments.keys())[0]
merged = merge_dicts(config, experiments[experiment_name]['config'])
merged["num_workers"] = 2
merged["num_envs_per_worker"] = 1
merged['exploration_config']['initial_epsilon'] = 0
merged['exploration_config']['final_epsilon'] = 0
if "use_pytorch" in experiments and experiments["use_pytorch"] == True:
    merged["framework"] = "torch"
    del merged["use_pytorch"]


agent = ApexTrainer(merged, FlatlandMinimalWrapper)
agent.restore("checkpoints/checkpoint_80/checkpoint-80")


obs = [0,0,0,0,0,0]
agent.compute_action(obs)
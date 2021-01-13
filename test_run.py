import argparse
import time
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.dqn import ApexTrainer
from envs.flatland_base import FlatlandBase
from ray.rllib.agents import dqn
from envs.flatland_random_sparse_small import FlatlandRandomSparseSmall
from envs.flatland.observations.custom_graph_obs import GraphObservartion, CustomGraphObservation, Features
from ray.rllib import MultiAgentEnv
import ray
import gym
from action_blocking_helping_functions import stop_deadlock_when_unavoidable
from action_blocking_helping_functions import reset_timestamp_dict
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, sparse_rail_generator, rail_from_file
import matplotlib.pyplot as plt
from flatland.utils.rendertools import RenderLocal
from flatland.envs.agent_utils import RailAgentStatus
import yaml
from flatland.envs.schedule_generators import sparse_schedule_generator, schedule_from_file
from flatland.envs.malfunction_generators import MalfunctionParameters, malfunction_from_params, ParamMalfunctionGen, \
    malfunction_from_file
from envs.flatland.utils.gym_env_wrappers import available_actions
from envs.flatland.observations.segment_graph import Graph

from utils.loader import load_envs, load_models
import os

import cv2



load_models(os.getcwd())

"""
USAGE:

python test_run.py --yaml recording/apex.yaml --checkpoint checkpoints/checkpoint_xx/checkpoint-xx
"""


def _transform_obs(rail_env, obs):
    return {
        agent_id_transform_obs: {
            'obs': agent_obs,
            'available_actions': np.asarray(available_actions(rail_env, rail_env.agents[agent_id_transform_obs], False))
        } for agent_id_transform_obs, agent_obs in obs.items()
    }


class FlatlandMinimalWrapper(gym.Env):
    def __init__(self, env):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(low=-1, high=np.inf, shape=(len(Features._fields),)),
            'available_actions': gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)
        })
        self._env = None

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()


parser = argparse.ArgumentParser()
parser.add_argument('--yaml', help='pass path to yaml file used for model training', type=str, required=True,
                    metavar='recording/apex.yaml')
parser.add_argument('--checkpoint', help='pass path to checkpoint', type=str, required=True,
                    metavar='checkpoints/checkpoint_xx/checkpoint-xx')
parser.add_argument('--plot', help='if you want to visualize with matplotlib', type=bool, default=False, const=True,
                    nargs='?')

args = parser.parse_args()
yaml_path = args.yaml
checkpoint_path = args.checkpoint
plot_flag = args.plot

with open(yaml_path) as f:
    experiments = yaml.safe_load(f)

ray.init(ignore_reinit_error=False, configure_logging=False)
config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
experiment_name = list(experiments.keys())[0]
merged = merge_dicts(config, experiments[experiment_name]['config'])
merged["num_workers"] = 0
merged["num_envs_per_worker"] = 0

if "use_pytorch" in experiments and experiments["use_pytorch"] == True:
    merged["framework"] = "torch"
    del merged["use_pytorch"]
# config['framework'] = 'torch'
merged['exploration_config']['initial_epsilon'] = 0
agent = ApexTrainer(merged, FlatlandMinimalWrapper)
agent.restore(checkpoint_path)

n_agents = 1
n_cities = 2
x_dim = 25
y_dim = 25
max_rails_in_city = 4
for i in range(30):

    x_dim = np.int(x_dim)
    y_dim = np.int(y_dim)
    n_agents = np.int(n_agents)
    n_cities = np.int(n_cities)

    print(f"n_agents: {n_agents}, x: {x_dim}, y:{y_dim},  cities: {n_cities}")

    obs_builder = CustomGraphObservation()
    speed_ration_map = {1.: 1.}  # Fast passenger train

    # We can now initiate the schedule generator with the given speed profiles

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    # We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
    # during an episode.

    stochastic_data = MalfunctionParameters(malfunction_rate=1 / 250,  # Rate of malfunction occurence
                                            min_duration=20,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

    env = RailEnv(width=x_dim, height=y_dim,
                  rail_generator=sparse_rail_generator(n_cities, False, 2, max_rails_in_city, seed=42),
                  number_of_agents=n_agents,
                  obs_builder_object=obs_builder, schedule_generator=schedule_generator)

    obs, info = env.reset(True, True, False)
    obs = _transform_obs(env, obs)
    env_renderer = RenderLocal(env, gl="PGL", agent_render_variant=4, show_debug=True)
    img = env_renderer.render_env_svg(show=False, show_observations=True, return_image=True)
    max_time_steps = int(4 * 2 * (env.width + env.height + n_agents / n_cities))
    # Print the observation vector for each agents
    out = cv2.VideoWriter(f'videos/recording_{i}.avi', cv2.VideoWriter_fourcc(*'XVID'), 13,
                          (img.shape[0], img.shape[1]))

    for _ in range(max_time_steps):
        if not plot_flag:
            img = env_renderer.render_env_svg(show=True, show_observations=True, return_image=True, show_rowcols=True)
        else:
            img = env_renderer.render_env_svg(show=False, show_observations=True, return_image=True, show_rowcols=True)
            # time.sleep(0.5)
            plt.imshow(img)
            plt.pause(0.001)
            plt.ion()
            plt.show()

        action_dict = {}
        num_active_agents = len(
            [i for i in range(env.number_of_agents) if env.agents[i].status == RailAgentStatus.ACTIVE])

        # print('='*30)
        # print(num_active_agents)
        # time.sleep(0.4)
        initial_pos_set = set()
        timestamp_segment_dict = dict()
        to_reset = []

        for agent_id in obs:
            agent_id_copy = agent_id
            #print(f"AGENT:{agent_id} ||deadlock_in_segment {obs[agent_id]['obs'][7]}|pdl {obs[agent_id]['obs'][19]}, pdf {obs[agent_id]['obs'][20]}, pdr {obs[agent_id]['obs'][21]}, segment_unusable {obs[agent_id]['obs'][36]}, | priority {obs[agent_id]['obs'][39]}")

            action = agent.compute_action(obs[agent_id_copy])
            print(f"action {action + 1} for agent {agent_id}")
            print("=" * 10)
            if num_active_agents < 500 and env.agents[agent_id].initial_position not in initial_pos_set and env.agents[
                agent_id].status == RailAgentStatus.READY_TO_DEPART:
                action += 1
                num_active_agents += 1
                initial_pos_set.add(env.agents[agent_id].initial_position)

            elif env.agents[agent_id].status == RailAgentStatus.ACTIVE:
                action += 1
                # if (obs[agent_id]['obs'][19] != 1 and obs[agent_id]['obs'][20] != 1 and obs[agent_id]['obs'][21] != 1 and obs[agent_id]['obs'][36] == 0):
                #
                #     # 10 11 12
                #
                #     if obs[agent_id]['obs'][10] < obs[agent_id]['obs'][11] and obs[agent_id]['obs'][10] < obs[agent_id]['obs'][12]:
                #         action = 1
                #     elif obs[agent_id]['obs'][11] < obs[agent_id]['obs'][10] and obs[agent_id]['obs'][11] < obs[agent_id]['obs'][12]:
                #         action = 2
                #     elif obs[agent_id]['obs'][12] < obs[agent_id]['obs'][11] and obs[agent_id]['obs'][12] < obs[agent_id]['obs'][10]:
                #         action = 3
                #     else:
                #         action = 2

            else:
                action = 4
            action_dict[agent_id] = action
            old_pos = Graph.get_virtual_position(agent_id)
            cell_transition = [i for i, v in enumerate(env.rail.get_transitions(*(old_pos[0], old_pos[1]),
                                                                                env.agents[agent_id].direction)) if
                               v == 1]

            if obs[agent_id]['obs'][30] == 1 and len(cell_transition) > 1 and action != 4:
                mask = [1 if obs[agent_id]['obs'][10] < 5000 else 0, 1 if obs[agent_id]['obs'][11] < 5000 else 0, 1 if
                obs[agent_id]['obs'][12] < 5000 else 0, 1]
                timestamp_segment_dict, to_reset, action_dict[agent_id] = \
                    stop_deadlock_when_unavoidable(timestamp_segment_dict=timestamp_segment_dict,
                                                   to_reset=to_reset,
                                                   handle=agent_id,
                                                   direction=env.agents[agent_id].direction,
                                                   action=action,
                                                   action_mask=mask,
                                                   old_pos=old_pos)

        obs, all_rewards, done, _ = env.step(action_dict)
        obs = _transform_obs(env, obs)

        timestamp_segment_dict = reset_timestamp_dict(timestamp_segment_dict, to_reset)
        to_reset = []

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
        if done["__all__"]:
            print("All agents reached target")
            break

    if not plot_flag:
        env_renderer.close_window()

    out.release()

    n_agents = n_agents + np.ceil(np.power(10, len(str(n_agents)) - 1) * 0.75)
    n_cities = (n_agents // 10) + 2
    x_dim = np.ceil(np.sqrt((2 * np.ceil(max_rails_in_city / 2.0 + 3)) ** 2 * (1.5 * n_cities))) + 7
    y_dim = x_dim


ray.shutdown()

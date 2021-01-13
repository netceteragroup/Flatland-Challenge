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
from envs.flatland.observations.segment_graph import Graph
from action_blocking_helping_functions import stop_deadlock_when_unavoidable
from action_blocking_helping_functions import reset_timestamp_dict
from flatland.evaluators.client import TimeoutException


load_models(os.getcwd())


def _transform_obs(rail_env, obs):
    return {
        agent_id: {
            'obs': agent_obs,
            'available_actions': np.asarray(available_actions(rail_env, rail_env.agents[agent_id], False))
        } for agent_id, agent_obs in obs.items()
    }


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


with open("apex_big.yaml") as f:
    experiments = yaml.safe_load(f)

ray.init()
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
# config['framework'] = 'torch'
agent = ApexTrainer(merged, FlatlandMinimalWrapper)
agent.restore("checkpoints/checkpoint_40_big/checkpoint-40")
#####################################################################
# Instantiate a Remote Client
#####################################################################
remote_client = FlatlandRemoteClient()

#####################################################################
# Define your custom controller
#
# which can take an observation, and the number of agents and
# compute the necessary action for this step for all (or even some)
# of the agents
#####################################################################


def my_controller(obs, number_of_agents, env, obs_builder):
    action_dict = {}
    obs = _transform_obs(env, obs)

    # initial_pos_set = set()
    # timestamp_segment_dict = dict()
    # to_reset = []
    # segment_direction_agents = {}
    # active_agents = [a.handle for a in env.agents if a.status == RailAgentStatus.ACTIVE]

    for agent_id in range(number_of_agents):
        if env.agents[agent_id].status == RailAgentStatus.DONE and env.agents[agent_id].status != RailAgentStatus.DONE_REMOVED:
            action = 4
        else:
            action = agent.compute_action(obs[agent_id]) + 1
        action_dict[agent_id] = action

    return action_dict

#####################################################################
# Instantiate your custom Observation Builder
#
# You can build your own Observation Builder by following
# the example here :
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14
#####################################################################
my_observation_builder = CustomGraphObservation()

# Or if you want to use your own approach to build the observation from the env_step,
# please feel free to pass a DummyObservationBuilder() object as mentioned below,
# and that will just return a placeholder True for all observation, and you
# can build your own Observation for all the agents as your please.
# my_observation_builder = DummyObservationBuilder()


#####################################################################
# Main evaluation loop
#
# This iterates over an arbitrary number of env evaluations
#####################################################################
evaluation_number = 0
while True:

    evaluation_number += 1
    # Switch to a new evaluation environemnt
    #
    # a remote_client.env_create is similar to instantiating a
    # RailEnv and then doing a env.reset()
    # hence it returns the first observation from the
    # env.reset()
    #
    # You can also pass your custom observation_builder object
    # to allow you to have as much control as you wish
    # over the observation of your choice.
    time_start = time.time()
    observation, info = remote_client.env_create(
        obs_builder_object=my_observation_builder
    )
    env_creation_time = time.time() - time_start
    if not observation:
        #
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been
        # evaluated on all the required evaluation environments,
        # and hence its safe to break out of the main evaluation loop
        break

    print("Evaluation Number : {}".format(evaluation_number))

    #####################################################################
    # Access to a local copy of the environment
    #
    #####################################################################
    # Note: You can access a local copy of the environment
    # by using :
    #       remote_client.env
    #
    # But please ensure to not make any changes (or perform any action) on
    # the local copy of the env, as then it will diverge from
    # the state of the remote copy of the env, and the observations and
    # rewards, etc will behave unexpectedly
    #
    # You can however probe the local_env instance to get any information
    # you need from the environment. It is a valid RailEnv instance.
    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    # Now we enter into another infinite loop where we
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    #
    # An episode is considered done when either all the agents have
    # reached their target destination
    # or when the number of time steps has exceed max_time_steps, which
    # is defined by :
    #
    # max_time_steps = int(4 * 2 * (env.width + env.height + 20))
    #
    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    while True:
        try:
            #####################################################################
            # Evaluation of a single episode
            #
            #####################################################################
            # Compute the action for this step by using the previously
            # defined controller
            time_start = time.time()
            action = my_controller(observation, number_of_agents, local_env, my_observation_builder)
            time_taken = time.time() - time_start
            time_taken_by_controller.append(time_taken)

            # Perform the chosen action on the environment.
            # The action gets applied to both the local and the remote copy
            # of the environment instance, and the observation is what is
            # returned by the local copy of the env, and the rewards, and done and info
            # are returned by the remote copy of the env
            time_start = time.time()
            observation, all_rewards, done, info = remote_client.env_step(action)
            steps += 1
            time_taken = time.time() - time_start
            time_taken_per_step.append(time_taken)

            if done['__all__']:
                print("Reward : ", sum(list(all_rewards.values())))
                #
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                break
        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive timeouts.
            print("Timeout! Will skip this episode and go to the next.", err)
            break

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("="*100)
    print("="*100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("="*100)

print("Evaluation of all environments complete...")
########################################################################
# Submit your Results
#
# Please do not forget to include this call, as this triggers the
# final computation of the score statistics, video generation, etc
# and is necesaary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())

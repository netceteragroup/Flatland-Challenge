import logging
from pprint import pprint

import gym
from flatland.envs.malfunction_generators import malfunction_from_params, no_malfunction_generator, MalfunctionParameters
# from flatland.envs.rail_env import RailEnv
from envs.flatland.utils.gym_env_wrappers import FlatlandRenderWrapper as RailEnv

from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file
from flatland.envs.schedule_generators import sparse_schedule_generator

from envs.flatland import get_generator_config
from envs.flatland.observations import make_obs
from envs.flatland.utils.gym_env import FlatlandGymEnv
from envs.flatland.utils.gym_env_wrappers import AvailableActionsWrapper, SkipNoChoiceCellsWrapper, SparseRewardWrapper, \
    DeadlockWrapper, ShortestPathActionWrapper, DeadlockResolutionWrapper, RewardWrapper
from envs.flatland.utils.schedule_generators import OurSchedGen
from envs.flatland_base import FlatlandBase
from random import randint, seed



class FlatlandSparse(FlatlandBase):

    def __init__(self, env_config) -> None:
        super().__init__()

        # TODO implement other generators
        assert env_config['generator'] == 'sparse_rail_generator'
        self._env_config = env_config

        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))

        seed()
        idx = randint(0, len(env_config['generator_config'])-1)
        # load random env specification
        self._name_cfg = env_config['generator_config'][idx]
        print("Loaded env:", self._name_cfg)

        self._config = get_generator_config(self._name_cfg)

        # Overwrites with env_config seed if it exists
        if env_config.get('seed'):
            self._config['seed'] = env_config.get('seed')

        if not hasattr(env_config, 'worker_index') or (env_config.worker_index == 0 and env_config.vector_index == 0):
            print("=" * 50)
            pprint(self._config)
            print("=" * 50)

        if env_config.get('custom_env', False):
            self._env = FlatlandGymEnv(
                rail_env=self._launch_custom_debugging(env_config.get('custom_env')),
                observation_space=self._observation.observation_space(),
                render=env_config.get('render'),
                regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
                regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
            )
        else:
            self._env = FlatlandGymEnv(
                rail_env=self._launch(),
                observation_space=self._observation.observation_space(),
                render=env_config.get('render'),
                regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
                regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
            )

        if env_config['observation'] == 'shortest_path':
            self._env = ShortestPathActionWrapper(self._env)
        if env_config['observation'] == 'dqn_shortest_path':
            self._env = ShortestPathActionWrapper(self._env)
        if env_config.get('reward_shaping', False):
            rewards = env_config.get("rewards")
            self._env = RewardWrapper(self._env, rewards)
        if env_config.get('sparse_reward', False):
            self._env = SparseRewardWrapper(self._env, finished_reward=env_config.get('done_reward', 1),
                                            not_finished_reward=env_config.get('not_finished_reward', -1))
        if env_config.get('deadlock_reward', 0) != 0:
            self._env = DeadlockWrapper(self._env, deadlock_reward=env_config['deadlock_reward'])
        if env_config.get('resolve_deadlocks', False):
            deadlock_reward = env_config.get('deadlock_reward', 0)
            self._env = DeadlockResolutionWrapper(self._env, deadlock_reward)
        if env_config.get('skip_no_choice_cells', False):
            self._env = SkipNoChoiceCellsWrapper(self._env, env_config.get('accumulate_skipped_rewards', False),
                                                 discounting=env_config.get('discounting', 1.))
        if env_config.get('available_actions_obs', False):
            self._env = AvailableActionsWrapper(self._env, env_config.get('allow_noop', True))

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    def _launch(self):
        rail_generator = sparse_rail_generator(
            seed=self._config['seed'],
            max_num_cities=self._config['max_num_cities'],
            grid_mode=self._config['grid_mode'],
            max_rails_between_cities=self._config['max_rails_between_cities'],
            max_rails_in_city=self._config['max_rails_in_city']
        )

        malfunction_generator = no_malfunction_generator()
        if {'malfunction_rate', 'malfunction_min_duration', 'malfunction_max_duration'} <= self._config.keys():
            # stochastic_data = {
            #     'malfunction_rate': self._config['malfunction_rate'],
            #     'min_duration': self._config['malfunction_min_duration'],
            #     'max_duration': self._config['malfunction_max_duration']
            # }
            if self._config['malfunction_rate'] > 0:
                malfunction_data = MalfunctionParameters(malfunction_rate= 1. / self._config['malfunction_rate'],
                                                     min_duration=self._config['malfunction_min_duration'],
                                                     max_duration=self._config['malfunction_max_duration'])
            else:
                malfunction_data = MalfunctionParameters(malfunction_rate=0,
                                                         min_duration=0,
                                                         max_duration=0)

            malfunction_generator = malfunction_from_params(malfunction_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = sparse_schedule_generator(speed_ratio_map)

        env = None
        try:
            env = RailEnv(
                width=self._config['width'],
                height=self._config['height'],
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=self._config['number_of_agents'],
                malfunction_generator_and_process_data=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=True,
                random_seed=self._config['seed'],
                # Should Below line be commented as here the env tries different configs,
                # hence opening it can be wasteful, morever the render has to be closed
                use_renderer=self._env_config.get('render')
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

    def _launch_custom_debugging(self, custom_env):
        rail_generator = rail_from_file(custom_env)

        malfunction_generator = no_malfunction_generator()
        if {'malfunction_rate', 'malfunction_min_duration', 'malfunction_max_duration'} <= self._config.keys():
            # stochastic_data = {
            #     'malfunction_rate': self._config['malfunction_rate'],
            #     'min_duration': self._config['malfunction_min_duration'],
            #     'max_duration': self._config['malfunction_max_duration']
            # }
            if self._config['malfunction_rate'] > 0:
                malfunction_data = MalfunctionParameters(malfunction_rate=1. / self._config['malfunction_rate'],
                                                         min_duration=self._config['malfunction_min_duration'],
                                                         max_duration=self._config['malfunction_max_duration'])
            else:
                malfunction_data = MalfunctionParameters(malfunction_rate=0,
                                                         min_duration=0,
                                                         max_duration=0)

            malfunction_generator = malfunction_from_params(malfunction_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = OurSchedGen()

        env = None
        try:
            env = RailEnv(
                width=self._config['width'],
                height=self._config['height'],
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=self._config['number_of_agents'],
                malfunction_generator_and_process_data=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=True,
                random_seed=self._config['seed'],
                # Should Below line be commented as here the env tries different configs,
                # hence opening it can be wasteful, morever the render has to be closed
                use_renderer=self._env_config.get('render')
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

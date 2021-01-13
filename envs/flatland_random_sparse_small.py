import random

import gym

from envs.flatland.utils.env_generators import random_sparse_env_small
from envs.flatland.observations import make_obs

from envs.flatland.utils.gym_env import FlatlandGymEnv
from envs.flatland.utils.gym_env_wrappers import SkipNoChoiceCellsWrapper, AvailableActionsWrapper, DeadlockWrapper, \
    SparseRewardWrapper, ShortestPathActionWrapper, DeadlockResolutionWrapper
from envs.flatland_base import FlatlandBase

class FlatlandRandomSparseSmall(FlatlandBase):

    def __init__(self, env_config) -> None:
        super().__init__()
        self._env_config = env_config
        self._test = env_config.get('test', False)
        self._min_seed = env_config['min_seed']
        self._max_seed = env_config['max_seed']
        assert self._min_seed <= self._max_seed
        self._min_test_seed = env_config.get('min_test_seed', 0)
        self._max_test_seed = env_config.get('max_test_seed', 100)
        assert self._min_test_seed <= self._max_test_seed
        self._next_test_seed = self._min_test_seed
        self._num_resets = 0
        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._env = FlatlandGymEnv(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            render=env_config.get('render'),
            regenerate_rail_on_reset=env_config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=env_config['regenerate_schedule_on_reset']
        )
        if env_config['observation'] == 'shortest_path':
            self._env = ShortestPathActionWrapper(self._env)
        if env_config.get('sparse_reward', False):
            self._env = SparseRewardWrapper(self._env, finished_reward=env_config.get('done_reward', 1),
                                            not_finished_reward=env_config.get('not_finished_reward', -1))
        if env_config.get('deadlock_reward', 0) != 0:
            self._env = DeadlockWrapper(self._env, deadlock_reward=env_config['deadlock_reward'])
        if env_config.get('resolve_deadlocks', False):
            deadlock_reward = env_config.get('deadlock_reward', 0)
            self._env = DeadlockResolutionWrapper(self._env, deadlock_reward)
        if env_config.get('skip_no_choice_cells', False):
            self._env = SkipNoChoiceCellsWrapper(self._env, env_config.get('accumulate_skipped_rewards', False))
        if env_config.get('available_actions_obs', False):
            self._env = AvailableActionsWrapper(self._env)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation.observation_space()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    def _generate_random_seed(self):
        random.seed(None)
        return random.randint(self._min_seed, self._max_seed)

    def _launch(self, max_tries=5):
        env = None
        num_tries = 0
        while env is None and num_tries < max_tries:
            if self._test:
                random_seed = self._next_test_seed
                rel_next_seed = self._next_test_seed - self._min_test_seed
                rel_max_seed = self._max_test_seed - self._min_test_seed
                self._next_test_seed = self._min_test_seed + ((rel_next_seed + 1) % (rel_max_seed + 1))  # inclusive max
            else:
                random_seed = self._generate_random_seed()
            random_seed = random_seed * 19997 + 997  # backwards consistency
            env = random_sparse_env_small(random_seed=random_seed, max_width=45, max_height=45,
                                          observation_builder=self._observation.builder())
            num_tries += 1
        if env is None:
            raise RuntimeError(f"Unable to launch env within {max_tries} tries.")
        return env

    def reset(self):
        if self._test or (
                self._env_config['reset_env_freq'] is not None
                and self._num_resets > 0
                and self._num_resets % self._env_config['reset_env_freq'] == 0
        ):
            self._env.env = self._launch()
        self._num_resets += 1
        return super().reset(random_seed=self._next_test_seed if self._test else self._generate_random_seed())

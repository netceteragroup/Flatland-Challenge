from typing import Any
from flatland.envs.schedule_utils import Schedule
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.schedule_generators import BaseSchedGen
from numpy.random.mtrand import RandomState
import numpy as np


class OurSchedGen(BaseSchedGen):
    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                 np_random: RandomState = None) -> Schedule:
        initial_pos = [(9,19) if i%2 == 0 else (6,5) for i in range(num_agents)]
        dest = [(4,4) if i%2 == 0 else (9,20) for i in range(num_agents)]
        agents_direction = [3 if i%2 == 0 else 1 for i in range(num_agents)]

        return Schedule(agent_positions=initial_pos, agent_directions=agents_direction,
                        agent_targets=dest, agent_speeds=np.ones(num_agents), agent_malfunction_rates=None,
                        max_episode_steps=400)


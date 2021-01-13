import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from envs.flatland.observations import Observation, register_obs


@register_obs("shortest_path")
class ShortestPathObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = ShortestPathForRailEnv(encode_one_hot=True)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=1, shape=(4,)),  # shortest path direction (one-hot)
            gym.spaces.Box(low=0, high=1, shape=(1,)),  # shortest path distance to target
            gym.spaces.Box(low=0, high=1, shape=(1,)),  # conflict when following shortest path (1=true, 0=false)
            gym.spaces.Box(low=0, high=1, shape=(4,)),  # other path direction (all zero if not available)
            gym.spaces.Box(low=0, high=1, shape=(1,)),  # other path direction (zero if not available)
            gym.spaces.Box(low=0, high=1, shape=(1,)),  # conflict when following other path  (1=true, 0=false)
        ])


class ShortestPathForRailEnv(ObservationBuilder):
    def __init__(self, encode_one_hot=True):
        super().__init__()
        self._encode_one_hot = encode_one_hot

    def reset(self):
        pass

    def get(self, handle: int = 0):
        self.env: RailEnv = self.env
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        directions = list(range(4))
        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        distance_map = self.env.distance_map.get()
        nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
        max_distance = np.max(distance_map[nan_inf_mask])
        assert not np.isnan(max_distance)
        assert max_distance != np.inf
        possible_steps = []

        # look in all directions for possible moves
        for movement in directions:
            if possible_transitions[movement]:
                next_move = movement
                pos = get_new_position(agent_virtual_position, movement)
                distance = distance_map[agent.handle][pos + (movement,)]  # new distance to target
                distance = max_distance if (distance == np.inf or np.isnan(distance)) else distance  # TODO: why does this happen?

                # look ahead if there is an agent between the agent and the next intersection
                # Todo: currently any train between the agent and the next intersection is reported. This includes
                # those that are moving away from the agent and therefore are not really conflicting. Will be improved.
                conflict = self.env.agent_positions[pos] != -1
                next_possible_moves = self.env.rail.get_transitions(*pos, movement)
                while np.count_nonzero(next_possible_moves) == 1 and not conflict:
                    movement = np.argmax(next_possible_moves)
                    pos = get_new_position(pos, movement)
                    conflict = self.env.agent_positions[pos] != -1
                    next_possible_moves = self.env.rail.get_transitions(*pos, movement)

                if self._encode_one_hot:
                    next_move_one_hot = np.zeros(len(directions))
                    next_move_one_hot[next_move] = 1
                    next_move = next_move_one_hot
                possible_steps.append((next_move, [distance/max_distance], [int(conflict)]))

        if len(possible_steps) == 1:
            # print(possible_steps[0] + (np.zeros(len(directions)), [.0], [0]))
            return possible_steps[0] + (np.zeros(len(directions)), [.0], [0])
        elif len(possible_steps) == 2:
            possible_steps = sorted(possible_steps, key=lambda step: step[1])  # sort by distance, ascending
            # print(possible_steps[0] + possible_steps[1])
            return possible_steps[0] + possible_steps[1]
        else:
            raise ValueError(f"More than two possibles steps at {agent_virtual_position}. Looks like a bug.")

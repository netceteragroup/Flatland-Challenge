import gym
from envs.flatland.observations import Observation, register_obs
"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict

import math
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from envs.flatland.utils.observation_utils import split_list_into_feature_groups, norm_obs_clip


@register_obs("dqn_shortest_path")
class DQNShortestPathWrapper(Observation):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = DQNShortestPath(max_depth=2)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(82,), dtype=np.float64)



Field = collections.namedtuple('Field', ['distance', 'action', 'position', 'direction'])


class DQNShortestPath(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """

    Node = collections.namedtuple('Node', 'dist_opposite_direction '
                                          'dist_same_direction '
                                          'potential_conflict '
                                          'dist_fastest_opposite_direction '
                                          'dist_slowest_same_directon '
                                          'malfunctioning_agent '
                                          'number_of_slower_agents_same_direction '
                                          'number_of_faster_agents_same_direction '
                                          'number_of_same_speed_agents_same_direction '
                                          'number_of_slower_agents_opposite_direction '
                                          'number_of_faster_agents_opposite_direction '
                                          'number_of_same_speed_agents_opposite_direction '
                                          'priority '
                                          'fastest_opposite_direction '
                                          'slowest_same_dir '
                                          'other_agent_wants_to_occupy_cell_in_path '
                                          'deadlock '
                                          'actions '
                                          'dist_own_target '
                                          'min_dist_others_target '
                                          'percentage_active_agents '
                                          'percentage_done_agents '
                                          'percentage_ready_to_depart_agents '
                                          'dist_usable_switch '
                                          'dist_unusable_switch '
                                          'currently_on_switch '
                                          'number_of_usable_switches_on_path '
                                          'number_of_unusable_switches_on_path '
                                          'mean_agent_speed_same_direction '
                                          'std_agent_speed_same_direction '
                                          'mean_agent_speed_diff_direction '
                                          'std_agent_speed_diff_direction '
                                          'mean_agent_malfunction_same_direction '
                                          'std_agent_malfunction_same_direction '
                                          'mean_agent_malfunction_diff_direction '
                                          'std_agent_malfunction_diff_direction '
                                          'sum_priorities_same_direction '
                                          'sum_priorities_diff_direction '
                                          'mean_agent_distance_to_target_same_direction '
                                          'std_agent_distance_to_target_same_direction '
                                          'mean_agent_distance_to_target_diff_direction '
                                          'std_agent_distance_to_target_diff_direction '
                                          'statistical_data_info '
                                          'path_distace'
                                  )

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    class Edge:
        """
          graph = {
              (pos_id1, dir1, pos_id2, dir2): {
                      'hash_position': int,
                      'switch_position_1': int,
                      'switch_id_1' : int,
                      'switch_position_2': int,
                      'switch_id_2' : int,
                      'length': int,
                      'neighbors': set{(pos_id1, dir1, pos_id2, dir2) ... }
                      'targets_for': set{handle1, handle2, handle3, ...}
                      'targets': {
                          handle1: {

                          },
                          handle2: {...}
                          ...
                      }
                      'observation': None or Node(...)
                  },
                  ...
          }
          """

        def __init__(self):
            self.switch_1_position = None
            self.switch_1_id = None
            self.switch_2_position = None
            self.switch_2_id = None
            self.length = 0
            self.neighbors = set()
            self.targets_for = set()
            self.targets = {}
            self.dist_to_target = {}
            self.observation = None

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None, how_many=2):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = len([f for f in DQNShortestPath.Node._fields if
                                    f not in ['actions', 'statistical_data_info', 'path_distace']])
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = {}
        self.HOW_MANY = how_many
        self.hash_position_direction_target = {}
        self.distance_map = None
        self.shortest_paths = {}
        self.obs_dict = {}
        self.hash_shortest_paths_agent = {}
        self.hash = {}
        self.hash_paths = {}
        self.prev_going = set([])
        self.set_ready_to_depart = set([])
        self.set_same_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_active = {}

        self.time_position = {}
        self.time_agent = {}
        self.speed_dict = {}
        self.agent_speed = {}

        self.time = []

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}
        self.hash_position_direction_target = {}
        self.shortest_hash = {}
        self.shortest_paths = {}
        self.hash_shortest_paths_agent = {}
        self.obs_dict = {}
        self.prev_going = set([])
        self.distance_map = None
        self.hash_paths = {}
        self.prev_going = set([])
        self.set_ready_to_depart = set([])
        self.set_same_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_active = {}

        self.time_position = {}
        self.time_agent = {}
        self.speed_dict = {}
        self.agent_speed = {}

    def is_switch_or_deadend(self, position):
        transition_bits = self.env.rail.grid[position[0]][position[1]]
        ret = bin(transition_bits).count('1') == 1
        return bool(sum([bin((transition_bits >> i) & 15).count('1') > 1 for i in [0, 4, 8, 16]])) or ret

    def explore(self, position, direction, targets_dict):
        """
          graph = {
              (pos_id1, dir1, pos_id2, dir2): {
                      'hash_position': int,
                      'switch_position_1': int,
                      'switch_id_1' : int,
                      'switch_position_2': int,
                      'switch_id_2' : int,
                      'length': int,
                      'neighbors': set{(pos_id1, dir1, pos_id2, dir2) ... }
                      'targets_for': set{handle1, handle2, handle3, ...}
                      'targets': {
                          handle1: {

                          },
                          handle2: {...}
                          ...
                      }
                      'observation': None or Node(...)
                  },
                  ...
          }

          edge, next_position, reverse_destination_direction
          """
        return_edge = self.Edge()
        return_edge.length = 0
        return_edge.targets_for = set()
        return_edge.dist_to_target = {}
        # valid_actions = get_valid_move_actions_(direction, position, self.env.rail)
        while True:
            if position in targets_dict:
                return_edge.targets_for.update(targets_dict[position])
                return_edge.dist_to_target[position] = return_edge.length
            valid_actions = list(get_valid_move_actions_(direction, position, self.env.rail))
            if len(valid_actions) > 1:
                break
            return_edge.length += 1
            _, position, direction = valid_actions[0]

        return return_edge, position, (direction + 2) % 4

    def reverse_edge(self, edge):
        pass

    def generate_graph(self):
        self.rail_graph = {}
        agents_per_target = {}

        for agent in self.env.agents:
            agents_per_target.setdefault(agent.target, [])
            agents_per_target[agent.target].append(agent.handle)

        visited = set()
        agents_position = [a.initial_position for a in self.env.agents]
        switches_to_explore = set()  # If two agents get to the same position we dont want the position twice thats why we save it to a set
        hash_edges = {}
        edges_of_position = {}

        for position in agents_position:  # Start from each agent positions and walk to the first switch, then add the switch into the switches_to_explore
            for i in range(4):
                valid_actions = list(get_valid_move_actions_(i, position,
                                                             self.env.rail))  # searching for direction with possible movements
                if len(valid_actions) > 0:
                    break
            action_, position, direction = valid_actions[0]
            while not self.is_switch_or_deadend(position):  # walk until a switch is found
                _, position, direction = list(get_valid_move_actions_(direction, position, self.env.rail))[0]
            switches_to_explore.add(position)

        # switches_to_explore = list(switches_to_explore)

        while len(switches_to_explore):
            position = switches_to_explore.pop()
            position_id = coordinate_to_position(self.env.width, [position])[0]  # dali raboti vaka?
            # switches_to_explore.add(position)
            valid_directions = [i for i in range(4) if len(get_valid_move_actions_(i, position,
                                                                                   self.env.rail)) > 0]  # ako nema dvizhenja vo taa direkcija dali vrakja []
            for direction in valid_directions:
                if position in hash_edges and direction in hash_edges[position]:
                    continue
                edge, next_position, reverse_destination_direction = self.explore(position, direction,
                                                                                  agents_per_target)

                if next_position not in visited and next_position is not position:
                    switches_to_explore.add(next_position)

                next_position_id = coordinate_to_position(self.env.width, [position])[0]

                if position_id > next_position_id:
                    position_id, next_position_id, position, next_position = next_position_id, position_id, next_position, position
                    direction, reverse_destination_direction = reverse_destination_direction, direction
                    edge = self.reverse_edge(edge)

                edges_of_position.setdefault(position,
                                             [])  # For each switch we keep his edges (gonna use it for finding the neighbours)
                edges_of_position.setdefault(next_position, [])
                edges_of_position[position].append(edge)
                edges_of_position[next_position].append(edge)

                edge.switch_1_position = position
                edge.switch_1_id = position_id
                edge.switch_2_position = next_position
                edge.switch_2_id = next_position_id
                edge.observation = None

                self.rail_graph[(position, direction, next_position, reverse_destination_direction)] = edge

                hash_edges.setdefault(position, {})
                hash_edges[position][direction] = self.rail_graph[
                    (position, direction, next_position, reverse_destination_direction)]
                # I think that the hash edges is unnecessary
                hash_edges.setdefault(next_position, {})
                hash_edges[next_position][reverse_destination_direction] = self.rail_graph[
                    (position, direction, next_position, reverse_destination_direction)]

            visited.add(position)
        for edge in self.rail_graph:
            pos_1 = self.rail_graph[edge].switch_1_position
            pos_2 = self.rail_graph[edge].switch_2_position

            self.rail_graph[edge].neighbors = list(set(edges_of_position[pos_1] + edges_of_position[pos_2]))
            self.rail_graph[edge].neighbors.remove(self.rail_graph[edge])

        self.rail_graph['hash_edges'] = hash_edges

    def choose_agent(self, speed_dict, set_free_path, set_in_the_same_group_as_prev_going, agent_speed, all_agents,
                     agent_with_same_end_and_start):
        '''
        #print(set_in_the_same_group_as_prev_going)
        if len(list(set_in_the_same_group_as_prev_going)) > 0:

            speed_pairs = [(agent_id, agent_speed[agent_id]) for agent_id in set_in_the_same_group_as_prev_going]
            sorted_speed_pairs = sorted(speed_pairs, key=lambda kv: (-kv[1], kv[0]))

            return sorted_speed_pairs[0][0]



        set_same_group_as_free = set([])
        for a_id in set_free_path:
            group_id = (self.env.agents[a_id].initial_position[0], self.env.agents[a_id].initial_position[1], self.env.agents[a_id].target[0], self.env.agents[a_id].target[1])
            for other_agent in agent_with_same_end_and_start[group_id]:
                if other_agent in all_agents:

                    set_same_group_as_free.add(other_agent)

        set_free_path=set_free_path.union(set_same_group_as_free)
        '''

        if len(list(set_free_path)) > 0:

            speed_pairs = [(agent_id, self.agent_speed[agent_id], agent_id) for agent_id in set_free_path]
            sorted_speed_pairs = sorted(speed_pairs, key=lambda kv: (-kv[1],kv[2]))
            # print("chosen",sorted_speed_pairs[0][0] )
            return sorted_speed_pairs[0][0]
        else:
            return None

    def set_group(self, _agent, dict_var):
        agent_set_start_end = (
            _agent.initial_position[0], _agent.initial_position[1], _agent.target[0], _agent.target[1],
            _agent.direction)
        dict_var.setdefault(agent_set_start_end, set([]))
        dict_var[agent_set_start_end].add(_agent.handle)

    def initialize_speed_dict(self):
        for a in self.env.agents:
            self.speed_dict.setdefault(a.speed_data['speed'], set([]))
            self.speed_dict[a.speed_data['speed']].add(a.handle)
            self.agent_speed[a.handle] = a.speed_data['speed']

    def initialize_groups(self, _agent):
        if _agent.status == RailAgentStatus.READY_TO_DEPART:
            self.set_group(_agent, self.agent_with_same_end_and_start_ready_to_depart)
            self.set_group(_agent, self.agent_with_same_end_and_start_active)
            self.set_ready_to_depart.add(_agent.handle)
            self.set_same_start_ready_to_depart.setdefault(_agent.initial_position, set([]))
            self.set_same_start_ready_to_depart[_agent.initial_position].add(_agent.handle)
        elif _agent.status == RailAgentStatus.ACTIVE:
            self.set_group(_agent, self.agent_with_same_end_and_start_active)
            # self.set_group(_agent, self.agent_with_same_end_and_start_ready_to_depart)
        else:
            self.prioritites[_agent.handle] = 1

    def build_time_idx_for_active_agents(self):
        self.time_position = {}
        self.time_agent = {}
        current_position_agent = {a.position: a.handle for a in self.env.agents if a.status == RailAgentStatus.ACTIVE}
        self.lies_on_path = {}
        self.positions = set([])

        for _agent in self.env.agents:
            if _agent.status != RailAgentStatus.ACTIVE:
                continue

            current_agent = _agent.handle

            # item = action, position, direction
            speed_agent = int(1 / _agent.speed_data['speed'])
            add_target = False
            prev_idx = -1
            for idx, item in enumerate(self.shortest_paths[current_agent][0]):
                add_target = True
                pos = item[1]
                if idx > 0:
                    self.positions.add(pos)

                if pos in current_position_agent and current_position_agent[pos] != _agent.handle:
                    self.lies_on_path.setdefault(_agent.handle, set([]))
                    self.lies_on_path[_agent.handle].add(current_position_agent[pos])

                position_fraction_ = self.env.agents[current_agent].speed_data['position_fraction']
                speed_ = self.env.agents[current_agent].speed_data['speed']

                cell_fraction_component = position_fraction_ / speed_
                if position_fraction_ >= 1:
                    cell_fraction_component = 0
                for i in range(speed_agent):
                    idx_for_time = int(idx * speed_agent + i - cell_fraction_component +
                                       _agent.malfunction_data['malfunction'])
                    if idx_for_time < 0:
                        continue
                    # assert prev_idx + 1 == idx_for_time
                    prev_idx += 1
                    self.time_position.setdefault(idx_for_time, {})
                    self.time_position[idx_for_time].setdefault(pos, set([]))
                    self.time_position[idx_for_time][pos].add(current_agent)

                    self.time_agent.setdefault(idx_for_time, {})
                    self.time_agent[idx_for_time][current_agent] = pos
            if add_target:
                if idx_for_time + 1 >= 0:
                    self.time_position.setdefault(idx_for_time + 1, {})
                    self.time_position[idx_for_time + 1].setdefault(self.env.agents[current_agent].target, set([]))
                    self.time_position[idx_for_time + 1][self.env.agents[current_agent].target].add(current_agent)

                    self.time_agent.setdefault(idx_for_time + 1, {})
                    self.time_agent[idx_for_time + 1][current_agent] = self.env.agents[current_agent].target

    def get_same_group_as_prev_going(self):
        set_agents_in_the_same_group_as_prev_going = set([])
        for a in self.prev_going:

            group_agent = self.env.agents[a]
            group = (group_agent.initial_position[0], group_agent.initial_position[1], group_agent.target[0],
                     group_agent.target[1], group_agent.direction)
            if group in self.agent_with_same_end_and_start_ready_to_depart:
                set_agents_in_the_same_group_as_prev_going = set_agents_in_the_same_group_as_prev_going.union(
                    self.agent_with_same_end_and_start_ready_to_depart[group])
        return set_agents_in_the_same_group_as_prev_going

    def does_agent_has_free_path(self, a_id, path=0):
        a = self.env.agents[a_id]
        free_path = True
        if a.initial_position in self.positions:
            free_path = False

        else:
            speed_agent = int(1 / a.speed_data['speed'])
            prev_pos = None
            if len(self.shortest_paths[a.handle]) <= path:
                return 0
            for idx, item in enumerate(self.shortest_paths[a.handle][path]):
                pos = item[1]

                for i in range(speed_agent):
                    time = idx * speed_agent + i + 1

                    if self.env.agents[a_id].status != RailAgentStatus.READY_TO_DEPART:
                        time=time - 1
                    if time in self.time_position and pos in self.time_position[time] and self.time_position[time][pos]:
                        l = len(self.time_position[time][pos])
                        if l > 2 or l == 1 and a_id not in self.time_position[time][pos]:
                            free_path = False
                            return free_path
                    if time - 1 in self.time_position and pos in self.time_position[time - 1]:
                        prev_agents = self.time_position[time - 1][pos]
                        for prev_a in prev_agents:
                            if prev_a != a_id and time in self.time_agent and prev_a in self.time_agent[time] \
                                    and self.time_agent[time][prev_a] == prev_pos:
                                free_path = False
                                return free_path
                prev_pos = pos
        return free_path

    def choose_agent_to_depart_for_group_flexible_free_path(self, group_depart,
                                                            set_agents_in_the_same_group_as_prev_going):

        set_free_path = set([])
        set_in_same_set_as_prev_going = set([])

        for group in self.agent_with_same_end_and_start_ready_to_depart:
            if group[0] == group_depart[0] and group[1] == group_depart[1]:
                first_agent = list(self.agent_with_same_end_and_start_ready_to_depart[group])[0]
                if self.does_agent_has_free_path(first_agent):
                    for agent in self.agent_with_same_end_and_start_ready_to_depart[group]:
                        set_free_path.add(agent)

                if first_agent in set_agents_in_the_same_group_as_prev_going:
                    for agent in self.agent_with_same_end_and_start_ready_to_depart[group]:
                        set_in_same_set_as_prev_going.add(agent)

        agent_id = self.choose_agent(set_free_path, set_in_same_set_as_prev_going,
                                     self.set_same_start_ready_to_depart[group_depart],
                                     self.agent_with_same_end_and_start_ready_to_depart)

        return agent_id

    def part_agents_by_speed(self, list_handles):
        speed_dict = {}
        for a_id in list_handles:
            a_speed = self.env.agents[a_id].speed_data['speed']
            speed_dict.setdefault(a_speed, set([]))
            speed_dict[a_speed].add(a_id)
        return speed_dict

    def choose_agent(self, set_free_path, set_in_the_same_group_as_prev_going, all_agents,
                     agent_with_same_end_and_start):
        if not self.agent_speed:
            self.initialize_speed_dict()
        '''
        #print(set_in_the_same_group_as_prev_going)
        if len(list(set_in_the_same_group_as_prev_going)) > 0:

            speed_pairs = [(agent_id, agent_speed[agent_id]) for agent_id in set_in_the_same_group_as_prev_going]
            sorted_speed_pairs = sorted(speed_pairs, key=lambda kv: (-kv[1], kv[0]))

            return sorted_speed_pairs[0][0]



        set_same_group_as_free = set([])
        for a_id in set_free_path:
            group_id = (self.env.agents[a_id].initial_position[0], self.env.agents[a_id].initial_position[1], self.env.agents[a_id].target[0], self.env.agents[a_id].target[1])
            for other_agent in agent_with_same_end_and_start[group_id]:
                if other_agent in all_agents:

                    set_same_group_as_free.add(other_agent)

        set_free_path=set_free_path.union(set_same_group_as_free)
        '''

        if len(list(set_free_path)) > 0:

            speed_pairs = [(agent_id, self.agent_speed[agent_id]) for agent_id in set_free_path]
            sorted_speed_pairs = sorted(speed_pairs, key=lambda kv: -kv[1])
            # print("chosen",sorted_speed_pairs[0][0] )
            return sorted_speed_pairs[0][0]
        else:
            return None

    def choose_agent_to_depart_for_group(self, group_depart, set_agents_in_the_same_group_as_prev_going):
        set_free_path = set([])
        set_in_same_set_as_prev_going = set([])

        for group in self.agent_with_same_end_and_start_ready_to_depart:

            if group[0] == group_depart[0] and group[1] == group_depart[1]:
                speed_dict = self.part_agents_by_speed(self.agent_with_same_end_and_start_ready_to_depart[group])

                for speed_group in speed_dict:
                    first_agent = list(speed_dict[speed_group])[0]
                    if self.does_agent_has_free_path(first_agent):
                        for agent in speed_dict[speed_group]:
                            set_free_path.add(agent)

                    if first_agent in set_agents_in_the_same_group_as_prev_going:
                        for agent in speed_dict[speed_group]:
                            set_in_same_set_as_prev_going.add(agent)

        agent_id = self.choose_agent(set_free_path, set_in_same_set_as_prev_going,
                                     self.set_same_start_ready_to_depart[group_depart],
                                     self.agent_with_same_end_and_start_ready_to_depart)

        return agent_id

    def add_starting_agent_to_time_idx(self, agent_id):
        speed_agent = int(1 / self.env.agents[agent_id].speed_data['speed'])
        add_target = False
        prev_tmp = 0
        for idx, item in enumerate(self.shortest_paths[agent_id][0]):
            add_target = True
            pos = item[1]
            self.positions.add(pos)

            for i in range(speed_agent):
                tmp = int(idx * speed_agent + i + 1)
                assert tmp == prev_tmp + 1
                prev_tmp += 1
                self.time_position.setdefault(tmp, {})
                self.time_position[tmp].setdefault(pos, set([]))
                self.time_position[tmp][pos].add(agent_id)

                self.time_agent.setdefault(tmp, {})
                self.time_agent[tmp][agent_id] = pos
        if add_target:
            pos = self.env.agents[agent_id].target
            self.time_position.setdefault(tmp + 1, {})
            self.time_position[tmp + 1].setdefault(pos, set([]))
            self.time_position[tmp + 1][pos].add(agent_id)

            self.time_agent.setdefault(tmp + 1, {})
            self.time_agent[tmp + 1][agent_id] = pos

    def build_graph(self, graph):
        self.time_position.setdefault(0, {})
        self.time_position.setdefault(-1, {})
        blocked_agents = set([])
        for idx in self.time_position:

            for pos in self.time_position[idx]:
                if idx - 1 in self.time_position:
                    if pos in self.time_position[idx - 1]:
                        for agent_prev in self.time_position[idx - 1][pos]:
                            for agent_now in self.time_position[idx][pos]:
                                if agent_prev == agent_now:
                                    continue
                                if agent_prev not in self.time_agent[idx]:
                                    continue
                                if agent_now in self.time_agent[idx - 1] and self.time_agent[idx - 1][agent_now] == \
                                        self.time_agent[idx][agent_prev]:
                                    # print("case 1", agent_prev, agent_now, pos, time_position[idx-1], time_position[idx])
                                    graph[agent_prev].add(agent_now)
                                    graph[agent_now].add(agent_prev)

                if len(self.time_position[idx][pos]) > 1:
                    for agent1 in self.time_position[idx][pos]:
                        for agent2 in self.time_position[idx][pos]:
                            if agent1 != agent2:
                                # print("case 2", time_position[idx] )
                                graph[agent1].add(agent2)
                                graph[agent2].add(agent1)
        return graph, blocked_agents

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """
        if self.distance_map is None:
            self.distance_map = self.env.distance_map.get()

        self.predicted_positions_of_moving_agents = {}
        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictor.set_env(self.env)
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        self.predicted_positions_of_moving_agents.setdefault(a, set([]))
                        self.predicted_positions_of_moving_agents[a].add(
                            coordinate_to_position(self.env.width, [list(self.predictions[a][t][1:3])])[0])

                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        # print("start")
        self.shortest_paths = {}
        self.prioritites = {}
        self.observation_hash = {}
        # ************************** PRIORITY ************************************
        self.shortest_paths = {}
        self.prioritites = {}
        self.priorities_second = {}

        graph = {}
        self.set_ready_to_depart = set([])
        self.set_same_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_ready_to_depart = {}
        self.agent_with_same_end_and_start_active = {}

        for _agent in self.env.agents:
            current_agent = _agent.handle

            if _agent.status in [RailAgentStatus.ACTIVE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = _agent
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']
                # self.location_has_target[tuple(_agent.position)] = _agent.handle
            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

            self.shortest_paths[current_agent] = self.get_2_dis_shortest_paths(current_agent)
            if self.shortest_paths[current_agent] is None:
                continue
            self.initialize_groups(_agent)
            if _agent.status == RailAgentStatus.ACTIVE:
                graph[current_agent] = set([])

        self.build_time_idx_for_active_agents()
        set_agents_in_the_same_group_as_prev_going = self.get_same_group_as_prev_going()

        for group_depart in self.set_same_start_ready_to_depart:
            agent_to_depart = self.choose_agent_to_depart_for_group(group_depart,
                                                                    set_agents_in_the_same_group_as_prev_going)
            if agent_to_depart is None:
                for other_agent_id in self.set_same_start_ready_to_depart[group_depart]:
                    self.prioritites[other_agent_id] = 0
                continue

            self.add_starting_agent_to_time_idx(agent_to_depart)
            graph[agent_to_depart] = set([])

            for other_agent_id in self.set_same_start_ready_to_depart[group_depart]:
                self.prioritites[other_agent_id] = 0

        graph, blocked_agents = self.build_graph(graph)

        # print("Graph", graph)
        # print("priorities", self.prioritites)

        components = self.connected_components(graph)
        # print("components,", components)
        for c in components:
            # print("c", c)
            colors = self.color_component(c, graph, self.prev_going, blocked_agents)
            # print("colors", colors)
            for a in colors:
                self.prioritites[a] = colors[a]

        self.priorities_second = {agent.handle: self.does_agent_has_free_path(agent.handle, 1) if agent.status in
                                                                                                  [RailAgentStatus.READY_TO_DEPART, RailAgentStatus.ACTIVE] else 0 for agent in
                                  self.env.agents }

        observations = super().get_many(handles)
        # print("end")
        return observations

    def color_component(self, component, graph, prev_going, blocked_agents):
        blocked_path_agents = set([])
        lies_on_path_component = {}
        for a in component:
            if a not in self.lies_on_path:
                continue
            in_same_component = [other for other in self.lies_on_path[a] if other in component]

            if len(in_same_component) > 0:
                lies_on_path_component[a] = set(in_same_component)
                blocked_path_agents = blocked_path_agents.union(in_same_component)

        set_component = set(component)
        non_blocked_agents = set_component.difference(lies_on_path_component.keys())
        # this is suspicios code
        non_blocked_agents = non_blocked_agents.intersection(blocked_path_agents)
        colors = {}
        tmp_colors = {}
        set_component = set(component)
        #        #teh, go with the ones that do not have blocked path

        for agent in non_blocked_agents:
            has_blocked_neighb = all([0 for neigh in graph[agent] if neigh in blocked_agents])
            if agent not in colors and agent in component and has_blocked_neighb:
                colors[agent] = 1
                for vertex in graph[agent]:
                    colors[vertex] = 0

        # first, go with non integer fractions
        for agent in component:
            has_blocked_neighb = all([0 for neigh in graph[agent] if neigh in blocked_agents])
            if self.env.agents[agent].speed_data['position_fraction'] > 0.0:
                if agent not in colors and agent in component and has_blocked_neighb:
                    colors[agent] = 1
                    for vertex in graph[agent]:
                        colors[vertex] = 0
        #   print("coloring", agent)

        # then, go prev_going
        # print(prev_going, "prev_going")
        elem_in_component = len(component)

        for agent in prev_going:
            if agent not in set_component:
                continue
            a = self.env.agents[agent]
            agent_group_id = (a.initial_position[0], a.initial_position[1], a.target[0], a.target[1], a.direction)

            for same_group_agent in self.agent_with_same_end_and_start_active[agent_group_id]:
                if same_group_agent in set_component:
                    has_blocked_neighb = all([0 for neigh in graph[same_group_agent] if neigh in blocked_agents])
                    if same_group_agent not in colors and agent and has_blocked_neighb:
                        colors[agent] = 1
                        for vertex in graph[agent]:
                            colors[vertex] = 0

        while len(colors.keys()) != elem_in_component:
            # print(colors)
            # print(elem_in_component)
            largest = None
            largest_value = -1
            for i in component:
                has_blocked_neighb = all([0 for neigh in graph[i] if neigh in blocked_agents])
                if i not in colors and len(graph[i]) > largest_value and has_blocked_neighb:
                    largest_value = len(graph[i])
                    largest = i
            if largest == None:
                break
            colors[largest] = 1

            for vertex in graph[largest]:
                colors[vertex] = 0
        for elem in component:
            colors.setdefault(elem, 0)
        for c in colors:
            if c in tmp_colors and colors[c] != tmp_colors[c]:
                print("DIFFERENCE FOR", c)
        return colors

    def connected_components(self, graph):
        visited = set([])

        list_of_groups = []
        while len(list(visited)) != len(list(graph.keys())):
            # choose unvisited node
            current_node = None
            for k in graph.keys():
                if k not in visited:
                    current_node = k
                    break
            current_stack = [current_node]
            this_group = []

            while len(current_stack) != 0:
                elem = current_stack.pop(0)
                visited.add(elem)
                this_group.append(elem)

                for neighbours in graph[elem]:
                    if neighbours not in visited and neighbours not in set(current_stack):
                        current_stack.append(neighbours)

            list_of_groups.append(this_group)
        return list_of_groups

    def bfs_shortest(self, position, direction, target, handle):
        if (position, direction, target) in self.shortest_hash:
            return self.shortest_hash[(position, direction, target)]

        distance = math.inf
        first_position = position
        first_direction = direction

        actions = []
        positions = [position]
        directions = [direction]

        while position != target:
            next_actions = get_valid_move_actions_(direction, position, self.env.rail)

            best_next_action = None
            for next_action in next_actions:
                next_action_distance = self.distance_map[
                    handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            if best_next_action is None:
                return

            position = best_next_action.next_position
            direction = best_next_action.next_direction

            actions.append(best_next_action.action)
            positions.append(best_next_action.next_position)
            directions.append(best_next_action.next_direction)

        paths_found = (actions, positions, directions)
        self.shortest_hash[(first_position, first_direction, target)] = paths_found
        return paths_found

    def BFS_v2(self, handle):
        agent = self.env.agents[handle]
        direction = agent.direction

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            return None
        start = agent_virtual_position
        target = agent.target
        if (start, direction, target) in self.hash_position_direction_target:
            return self.hash_position_direction_target[(start, direction, target)]

        distance = math.inf
        position = agent_virtual_position
        actions = []
        positions = [position]
        directions = [direction]

        while position != agent.target:
            next_actions = get_valid_move_actions_(direction, position, self.env.rail)
            distances_list = [self.distance_map[
                                  agent.handle, x.next_position[0], x.next_position[
                                      1], x.next_direction] for x in next_actions]
            if len(next_actions) > 1 and max(distances_list) < math.inf:

                assert len(next_actions) == 2

                next_actions = list(next_actions)
                next_actions.sort(
                    key=lambda x: self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction])
                # Here we found a switch

                first_path = self.bfs_shortest(next_actions[0].next_position, next_actions[0].next_direction,
                                               agent.target, handle)
                second_path = self.bfs_shortest(next_actions[1].next_position, next_actions[1].next_direction,
                                                agent.target, handle)

                first_actions = actions[:]
                first_positions = positions[:]
                first_directions = directions[:]

                first_actions.append(next_actions[0].action)
                first_actions.extend(first_path[0])
                first_positions.extend(first_path[1])
                first_directions.extend(first_path[2])

                actions.append(next_actions[1].action)
                actions.extend(second_path[0])
                positions.extend(second_path[1])
                directions.extend(second_path[2])

                paths_found = []
                paths_found.append(list(zip(first_actions, first_positions, first_directions)))
                paths_found.append(list(zip(actions, positions, directions)))

                self.hash_position_direction_target[(agent_virtual_position, agent.direction, target)] = paths_found
                return paths_found

            else:
                best_next_action = None
                for i, next_action in enumerate(next_actions):
                    next_action_distance = distances_list[i]
                    if next_action_distance < distance:
                        best_next_action = next_action
                        distance = next_action_distance

                if best_next_action is None:
                    return

                position = best_next_action.next_position
                direction = best_next_action.next_direction

                actions.append(best_next_action.action)
                positions.append(best_next_action.next_position)
                directions.append(best_next_action.next_direction)

        paths_found = []

        paths_found.append(list(zip(actions, positions, directions)))
        self.hash_position_direction_target[(agent_virtual_position, agent.direction, target)] = paths_found
        return paths_found

    def BFS(self, handle):
        visited = set()
        # visited = collections.defaultdict(lambda: self.HOW_MANY)
        queue = []
        agent = self.env.agents[handle]
        direction = agent.direction
        agent_first_direction = agent.direction

        distance_map = self.env.distance_map

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            return None
        start = agent_virtual_position
        target = agent.target
        if self.hash_position_direction_target.get(start, False) and self.hash_position_direction_target[start].get(
                direction, False) and self.hash_position_direction_target[start][direction].get(
            target, False):
            return self.hash_position_direction_target[start][direction][target]

        # visited[(start, direction)] -= 1
        visited.add((start, direction))
        queue.append((start, direction, 0, [], [start], [direction]))

        paths_found = []
        number_of_paths_found = 0
        max_steps = int((self.env.width + self.env.height + 20) * 8)
        while queue:
            current_position, direction, steps, actions, positions, directions = queue.pop(0)
            if steps > max_steps:
                break
            if current_position == agent.target:
                max_steps = min(max_steps, int(steps * 1.3))
                paths_found.append(list(zip(actions, positions, directions)))
                number_of_paths_found += 1
                if number_of_paths_found == self.HOW_MANY:
                    return paths_found
                continue

            next_actions = get_valid_move_actions_(direction, current_position, distance_map.rail)
            for action, new_position, new_direction in next_actions:
                # if visited[(new_position, new_direction)] > 0:
                if (new_position, new_direction) not in visited:
                    actions_copy = actions[:]
                    actions_copy.extend([action])

                    positions_copy = positions[:]
                    positions_copy.append(new_position)

                    directions_copy = directions[:]
                    directions_copy.append(new_direction)

                    # visited[(new_position, new_direction)] -= 1
                    visited.add((new_position, new_direction))
                    queue.append(
                        (new_position, new_direction, steps + 1, actions_copy, positions_copy, directions_copy))

        self.hash_position_direction_target.setdefault(start, {})
        self.hash_position_direction_target[start].setdefault(agent.direction, {})
        self.hash_position_direction_target[start][agent.direction][target] = paths_found
        return paths_found

    def BFS_2_shortest_helper(self, handle, start_position, first_direction, target):
        visited = set()

        queue = []
        distance_map = self.env.distance_map
        direction = first_direction
        start = start_position
        target = target

        if self.shortest_hash.get(start, False) and self.shortest_hash[start].get(
                direction, False) and self.shortest_hash[start][direction].get(
            target, False):
            return self.shortest_hash[start][direction][target]

        # visited[(start, direction)] -= 1
        visited.add((start, direction))
        queue.append((start, direction, 0, [], [start], [direction]))

        paths_found = []
        max_steps = int((self.env.width + self.env.height + 20) * 8)
        while queue:
            current_position, direction, steps, actions, positions, directions = queue.pop(0)
            if steps > max_steps:
                break
            if current_position == target:
                paths_found.append([actions, positions, directions])
                self.shortest_hash.setdefault(start_position, {})
                self.shortest_hash[start_position].setdefault(first_direction, {})
                self.shortest_hash[start_position][first_direction][target] = paths_found
                return paths_found

            next_actions = get_valid_move_actions_(direction, current_position, distance_map.rail)

            for action, new_position, new_direction in next_actions:
                # if visited[(new_position, new_direction)] > 0:
                if (new_position, new_direction) not in visited:
                    actions_copy = actions[:]
                    actions_copy.extend([action])

                    positions_copy = positions[:]
                    positions_copy.append(new_position)

                    directions_copy = directions[:]
                    directions_copy.append(new_direction)

                    # visited[(new_position, new_direction)] -= 1
                    visited.add((new_position, new_direction))
                    queue.append(
                        (new_position, new_direction, steps + 1, actions_copy, positions_copy, directions_copy))

        self.shortest_hash.setdefault(start_position, {})
        self.shortest_hash[start_position].setdefault(first_direction, {})
        self.shortest_hash[start_position][first_direction][target] = None
        return None

    def BFS_2_shortest(self, handle):
        visited = set()

        queue = []
        agent = self.env.agents[handle]
        direction = agent.direction

        distance_map = self.env.distance_map

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            return None
        start = agent_virtual_position
        target = agent.target
        if self.hash_position_direction_target.get(start, False) and self.hash_position_direction_target[start].get(
                direction, False) and self.hash_position_direction_target[start][direction].get(
            target, False):
            return self.hash_position_direction_target[start][direction][target]

        # visited[(start, direction)] -= 1
        visited.add((start, direction))
        queue.append((start, direction, 0, [], [start], [direction]))

        paths_found = []
        max_steps = int((self.env.width + self.env.height + 20) * 8)
        while queue:
            current_position, direction, steps, actions, positions, directions = queue.pop(0)
            if steps > max_steps:
                break
            if current_position == agent.target:
                paths_found.append(list(zip(actions, positions, directions)))
                paths_found.append(None)

                self.hash_position_direction_target.setdefault(start, {})
                self.hash_position_direction_target[start].setdefault(agent.direction, {})
                self.hash_position_direction_target[start][agent.direction][target] = paths_found
                return paths_found

            next_actions = get_valid_move_actions_(direction, current_position, distance_map.rail)
            # print("next actions", next_actions)
            if len(next_actions) > 1:

                k = len(positions)

                actions_copy = actions[:]
                positions_copy = positions[:]
                directions_copy = directions[:]
                next_actions = list(next_actions)
                path_left = self.BFS_2_shortest_helper(handle, next_actions[0].next_position,
                                                       next_actions[0].next_direction, agent.target)
                actions_copy.append(next_actions[0].action)
                if path_left is not None:
                    actions_copy.extend(path_left[0][0])
                    positions_copy.extend(path_left[0][1])
                    directions_copy.extend(path_left[0][2])
                    paths_found.append(list(zip(actions_copy, positions_copy, directions_copy)))
                else:
                    paths_found.append(None)

                path_right = self.BFS_2_shortest_helper(handle, next_actions[1].next_position,
                                                        next_actions[1].next_direction, agent.target)
                actions.append(next_actions[1].action)
                if path_right is not None:
                    actions.extend(path_left[0][0])
                    positions.extend(path_left[0][1])
                    directions.extend(path_left[0][2])
                    paths_found.append(list(zip(actions, positions, directions)))
                else:
                    paths_found.append(None)
                for k_idx in range(k):
                    start_for_hashing = paths_found[k][1]
                    direction_for_hashing = paths_found[k][2]
                    self.hash_position_direction_target.setdefault(start_for_hashing, {})
                    self.hash_position_direction_target[start_for_hashing].setdefault(direction_for_hashing, {})
                    self.hash_position_direction_target[start_for_hashing][direction_for_hashing][target] = paths_found[
                                                                                                            k:]
                return paths_found


            else:
                for action, new_position, new_direction in next_actions:
                    if (new_position, new_direction) not in visited:
                        actions_copy = actions[:]
                        actions_copy.extend([action])

                        positions_copy = positions[:]
                        positions_copy.append(new_position)

                        directions_copy = directions[:]
                        directions_copy.append(new_direction)

                        # visited[(new_position, new_direction)] -= 1
                        visited.add((new_position, new_direction))
                        queue.append(
                            (new_position, new_direction, steps + 1, actions_copy, positions_copy, directions_copy))

        self.hash_position_direction_target.setdefault(start, {})
        self.hash_position_direction_target[start].setdefault(agent.direction, {})
        self.hash_position_direction_target[start][agent.direction][target] = None
        return None

    def get_2_dis_shortest_paths_refactor(self, handle):
        # Field = collections.namedtuple('Field', ['distance', 'action', 'position', 'direction'])
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            initial_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            initial_position = agent.position
        else:
            return None

        direction = agent.direction
        direction_for_hashing = direction
        position_for_hashing = initial_position
        position = initial_position
        target = agent.target

        if (direction, position, target) in self.hash_position_direction_target:
            paths_found = self.hash_position_direction_target[(direction, position, target)]
            self.hash_shortest_paths_agent[handle] = paths_found
            # why is the second hash needed???
            return paths_found

        valid_actions = list(get_valid_move_actions_(direction, position, self.env.rail))

        list_initial_actions = []
        list_initial_positions = [position]
        list_initial_directions = [direction]

        while (len(valid_actions) <= 1 or max(
                [self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction] for x in
                 valid_actions]) == np.inf) and position != target:
            if max([self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction] for x in
                    valid_actions]) == np.inf:
                valid_actions.sort(
                    key=lambda x: self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction])

            action, new_position, new_direction = valid_actions[0]

            list_initial_actions.append(action)
            list_initial_positions.append(new_position)
            list_initial_directions.append(new_direction)

            position, direction = new_position, new_direction
            valid_actions = list(get_valid_move_actions_(direction, position, self.env.rail))

        if position == target:
            paths_found = [list(zip(list_initial_actions, list_initial_positions, list_initial_directions))]
            self.hash_shortest_paths_agent[handle] = paths_found
            return paths_found

        valid_actions.sort(
            key=lambda x: self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction])

        paths_found = []
        for v in valid_actions:
            list_actions = list_initial_actions[:]
            list_positions = list_initial_positions[:]
            list_directions = list_initial_directions[:]

            list_actions.append(v.action)
            list_next_actions, list_next_positions, list_next_directions = self.get_shortest_path(v.next_position,
                                                                                                  v.next_direction,
                                                                                                  handle)
            list_actions.extend(list_next_actions)
            list_positions.extend(list_next_positions)
            list_directions.extend(list_next_directions)

            paths_found.append(list(zip(list_actions, list_positions, list_directions)))

        self.hash_position_direction_target[(direction_for_hashing, position_for_hashing, target)] = paths_found
        self.hash_shortest_paths_agent[handle] = paths_found

        return paths_found

    def is_switch(self, position, direction):
        transition_bits = self.env.rail.grid[position[0]][position[1]]
        return len(get_valid_move_actions_(direction, position, self.env.rail)) > 1 or bin(transition_bits).count(
            '1') == 1

    def get_2_dis_shortest_paths(self, handle):
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            initial_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            initial_position = agent.position
        else:
            return None

        initial_direction = agent.direction
        target = agent.target

        position, direction = initial_position, initial_direction

        if self.hash_position_direction_target.get(initial_position, False) and self.hash_position_direction_target[
            initial_position].get(initial_direction, False) and \
                self.hash_position_direction_target[initial_position][initial_direction].get(target, False):
            paths_found = self.hash_position_direction_target[initial_position][initial_direction][target]
            self.hash_shortest_paths_agent[handle] = paths_found
            return paths_found

        valid_actions = list(get_valid_move_actions_(direction, position, self.env.rail))
        prev_valid_actions = list(
            get_valid_move_actions_(agent.old_direction if agent.old_direction is not None else initial_direction,
                                    agent.old_position if agent.old_position is not None else initial_position,
                                    self.env.rail))

        if handle in self.hash_shortest_paths_agent and len(prev_valid_actions) != 2:
            paths = self.hash_shortest_paths_agent[handle]
            truncated_paths = []

            for i, path in enumerate(paths):
                index_ = [field[1] for field in path].index(position)
                truncated_paths.append(path[index_:])
            return truncated_paths

        list_initial_actions = []
        list_initial_positions = [position]
        list_initial_directions = [direction]
        list_initial_is_switch = [self.is_switch(position, direction)]

        while (len(valid_actions) <= 1 or max(
                [self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction] for x in
                 valid_actions]) == np.inf) and position != target:
            if max([self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction] for x in
                    valid_actions]) == np.inf:
                valid_actions.sort(
                    key=lambda x: self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction])

            action, new_position, new_direction = valid_actions[0]

            try:
                list_initial_actions.append(action)
                list_initial_positions.append(new_position)
                list_initial_directions.append(new_direction)
                list_initial_is_switch.append(self.is_switch(new_position, new_direction))
            except MemoryError as memory_error:
                print(memory_error)
                print('handle', handle, 'initial_position', initial_position, 'initial_direction', initial_direction)
                print('Distance map: ')
                print(self.distance_map[handle][new_position[0]][new_position[1]][new_direction])
                return None

            position, direction = new_position, new_direction
            valid_actions = list(get_valid_move_actions_(direction, position, self.env.rail))

        if position == target:
            paths_found = [list(
                zip(list_initial_actions, list_initial_positions, list_initial_directions, list_initial_is_switch))]
            self.hash_shortest_paths_agent[handle] = paths_found
            return paths_found

        valid_actions.sort(
            key=lambda x: self.distance_map[handle][x.next_position[0]][x.next_position[1]][x.next_direction])

        paths_found = []
        for v in valid_actions:
            list_actions = list_initial_actions[:]
            list_positions = list_initial_positions[:]
            list_directions = list_initial_directions[:]
            list_is_switch = list_initial_is_switch[:]

            list_actions.append(v.action)
            list_next_actions, list_next_positions, list_next_directions, list_next_is_switch = self.get_shortest_path(
                v.next_position,
                v.next_direction,
                handle)
            list_actions.extend(list_next_actions)
            list_positions.extend(list_next_positions)
            list_directions.extend(list_next_directions)
            list_is_switch.extend(list_next_is_switch)

            paths_found.append(list(zip(list_actions, list_positions, list_directions, list_is_switch)))

        self.hash_position_direction_target.setdefault(initial_position, {})
        self.hash_position_direction_target[initial_position].setdefault(initial_direction, {})
        self.hash_position_direction_target[initial_position][initial_direction][target] = paths_found

        self.hash_shortest_paths_agent[handle] = paths_found

        return paths_found

    def get_shortest_path(self, position, direction, handle):
        list_actions = []
        list_positions = [position]
        list_directions = [direction]
        list_is_switch = [self.is_switch(position, direction)]

        target = self.env.agents[handle].target

        while position != target:
            best_field = None
            prev_min = np.inf
            valid_actions = get_valid_move_actions_(direction, position, self.env.rail)
            for action, new_position, new_direction in valid_actions:
                x_, y_ = new_position
                distance = self.distance_map[handle][x_][y_][new_direction]
                if distance < prev_min:
                    best_field = Field(distance, action, new_position, new_direction)
                    prev_min = distance

            list_actions.append(best_field.action)
            list_positions.append(best_field.position)
            list_directions.append(best_field.direction)
            list_is_switch.append(self.is_switch(best_field.position, best_field.direction))

            position, direction = best_field.position, best_field.direction
        return list_actions, list_positions, list_directions, list_is_switch

    def get_k_shortest_refactor(self, handle):
        # Not working!!
        agent = self.env.agents[handle]
        distance_map = self.distance_map

        initial_path = []

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            initial_position = agent.initial_position
            initial_path.append((RailEnvActions.MOVE_FORWARD, initial_position))
        elif agent.status == RailAgentStatus.ACTIVE:
            initial_position = agent.position
        else:
            return None

        target = agent.target

        position = initial_position
        direction = agent.direction  # ??
        initial_direction = agent.direction

        visited = set()
        paths_found = []

        if self.hash_position_direction_target.get(initial_position, False) and self.hash_position_direction_target[
            initial_position].get(direction, False) and self.hash_position_direction_target[initial_position][
            direction].get(
            target, False):
            return self.hash_position_direction_target[initial_position][direction][target]

        for _ in range(self.HOW_MANY):
            path = initial_path[:]

            while position != target:
                valid_actions = get_valid_move_actions_(direction, position, self.env.rail)
                fields = []
                for action, new_position, new_direction in valid_actions:
                    x_, y_ = new_position
                    distance = distance_map[handle][x_][y_][new_direction]
                    next_field = Field(distance, action, new_position, new_direction)
                    if next_field in visited:
                        continue
                    fields.append(next_field)

                fields.sort(key=lambda x: x.distance)
                best_field = fields.pop(0)
                path.append((best_field.action, position))
                if len(valid_actions) > 1:
                    visited.add(best_field)
                position, direction = best_field.position, best_field.direction
            paths_found.append(path)

        self.hash_position_direction_target[initial_position] = {}
        self.hash_position_direction_target[initial_position][initial_direction] = {}
        self.hash_position_direction_target[initial_position][initial_direction][target] = paths_found
        return paths_found

    def get_k_shortest(self, handle):
        agent = self.env.agents[handle]
        distance_map = self.distance_map

        path = []

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            initial_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            initial_position = agent.position
        else:
            return None

        initial_direction = agent.direction
        paths_found = []

        switches = []
        target = agent.target

        if self.hash_position_direction_target.get(initial_position, False) and self.hash_position_direction_target[
            initial_position].get(initial_direction, False) and self.hash_position_direction_target[initial_position][
            initial_direction].get(
            target, False):
            return self.hash_position_direction_target[initial_position][initial_direction][target]

        for i in range(self.HOW_MANY):
            if i is not 0 and len(switches) is 0:
                return paths_found

            position = initial_position
            direction = initial_direction
            switches.sort(key=lambda x: x['distance_target'])

            curr_distance = 0
            if i is not 0:
                if len(switches) is 0:
                    return paths_found
                best_act = switches.pop(0)
                curr_distance = best_act['curr_distance'] + 1
                position, direction = best_act['position'], best_act['direction']
                path = paths_found[best_act['father_path']][:best_act['from']]
                path.append((best_act['action'], position, direction))

            while True:
                valid_actions = get_valid_move_actions_(direction, position, self.env.rail)
                possible_mov = []

                for action, new_position, new_direction in valid_actions:
                    x_, y_ = new_position
                    distance = distance_map[handle][x_][y_][new_direction]
                    possible_mov.append({
                        'distance_target': distance + curr_distance,
                        'curr_distance': curr_distance,
                        'direction': new_direction,
                        'position': new_position,
                        'father_path': i,
                        'from': len(path),
                        'action': action
                    })
                possible_mov.sort(key=lambda x: x['distance_target'])
                best_act = possible_mov.pop(0)
                path.append((best_act['action'], position, direction))
                switches.extend(possible_mov)
                if position == agent.target:
                    break
                position, direction = best_act['position'], best_act['direction']
                curr_distance += 1

            paths_found.append(path)

        self.hash_position_direction_target[initial_position] = {}
        self.hash_position_direction_target[initial_position][initial_direction] = {}
        self.hash_position_direction_target[initial_position][initial_direction][target] = paths_found
        return paths_found

    def process_custom_observation(self, obs):
        processed_observation = dict()
        num_features_per_node = self.observation_dim
        data, number, bol_data, speed = split_list_into_feature_groups(obs)

        processed_observation["data"] = data
        processed_observation["number"] = number
        processed_observation["bol_data"] = bol_data
        processed_observation["speed"] = speed
        processed_observation["deadlock"] = True
        for item in obs:
            if item != None:
                processed_observation["deadlock"] = processed_observation["deadlock"] and item.deadlock

        data = norm_obs_clip(data)[:]
        number = norm_obs_clip(number)[:]
        bol_data = norm_obs_clip(bol_data)[:]
        speed = norm_obs_clip(speed)[:]
        normalized_obs = np.concatenate((np.concatenate((np.concatenate((data, number)), bol_data)), speed))
        processed_observation["agent_state"] = normalized_obs[:]
        # processed_observation["station_distance"] = distance[0]
        # processed_observation["deadlock"] = agent_data[2]
        return processed_observation['agent_state'].astype(np.float64)

    def get(self, handle: int = 0) -> Node:
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            #None
            return self.process_custom_observation([None, None])
        if self.shortest_paths[handle] is None:
            #return None
            return self.process_custom_observation([None, None])

        self.obs_dict[handle] = []
        if agent.speed_data['position_fraction'] == 0.0:
            for idx, path in enumerate(self.shortest_paths[handle]):
                if idx > 0:
                    priority = self.priorities_second[handle]
                else:
                    priority = self.prioritites[handle]
                observation_for_path = self._explore_path(handle, path, priority)
                self.obs_dict[handle].append(observation_for_path)

        found_paths = len(self.obs_dict[handle])
        for i in range(self.HOW_MANY - found_paths):
            self.obs_dict[handle].append(None)

        # return self.obs_dict([handle])
        return self.process_custom_observation(self.obs_dict[handle])

    def explore_path(self, handle, list_walking_elements, priority):
        # FIXME: If len(list_walking_elements) < 2 then _explore_path with all list_walking_elements

        actions = []

        dist_unusable_switch = np.inf
        dist_opposite_direction = np.inf
        dist_same_direction = np.inf
        number_of_slower_agents_same_direction = 0
        number_of_faster_agents_same_direction = 0
        number_of_same_speed_agents_same_direction = 0
        number_of_slower_agents_opposite_direction = 0
        number_of_faster_agents_opposite_direction = 0
        number_of_same_speed_agents_opposite_direction = 0
        min_dist_others_target = np.inf
        potential_conflict = np.inf
        other_agent_wants_to_occupy_cell_in_path = False

        dist_fastest_opposite_direction = np.inf
        dist_slowest_same_directon = np.inf

        malfunctioning_agent = 0
        fastest_opposite_direction = 0
        slowest_same_dir = np.inf

        dist_usable_switch = -1
        currently_on_switch = 0
        number_of_usable_switches_on_path = sum([int(e[3]) for e in list_walking_elements])
        number_of_unusable_switches_on_path = 0

        percentage_active_agents = -1
        percentage_done_agents = -1
        percentage_ready_to_depart_agents = -1

        mean_agent_speed_same_direction = 0
        std_agent_speed_same_direction = 0
        mean_agent_speed_diff_direction = 0
        std_agent_speed_diff_direction = 0
        mean_agent_malfunction_same_direction = 0
        std_agent_malfunction_same_direction = 0
        mean_agent_malfunction_diff_direction = 0
        std_agent_malfunction_diff_direction = 0
        sum_priorities_same_direction = 0
        sum_priorities_diff_direction = 0

        mean_agent_distance_to_target_same_direction = 0
        std_agent_distance_to_target_same_direction = 0
        mean_agent_distance_to_target_diff_direction = 0
        std_agent_distance_to_target_diff_direction = 0

        speeds_same_direction = []
        speeds_diff_direction = []

        malfunctions_same_direction = []
        malfunctions_diff_direction = []

        distances_to_target_same_direction = []
        distances_to_target_diff_direction = []

        ###################### New default params ######################
        deadlock = False  # False by default
        dist_own_target = 0
        path_distance = 0

        switches_indexes = [i for i, e in enumerate(list_walking_elements) if
                            e[3] or i is 0 or i is len(list_walking_elements) - 1]

        for i in range(len(switches_indexes) - 1):
            start_action, start_position, start_direction, _ = list_walking_elements[switches_indexes[i]]
            end_action, end_position, end_direction, _ = list_walking_elements[switches_indexes[i + 1]]
            to_target = False
            if i < len(switches_indexes) - 2:
                branch = list_walking_elements[switches_indexes[i]: switches_indexes[i + 1]]
            else:
                branch = list_walking_elements[switches_indexes[i]:]
                to_target = True

            hash_key = (start_position, start_direction, start_action, end_position, to_target)

            if hash_key not in self.observation_hash:
                self.observation_hash[hash_key] = self._explore_path(handle, branch, priority)

            temp_node: DQNShortestPath.Node = self.observation_hash[hash_key]

            dist_opposite_direction = min(dist_opposite_direction, temp_node.dist_opposite_direction)
            dist_same_direction = min(dist_same_direction, temp_node.dist_same_direction)
            potential_conflict = min(potential_conflict, temp_node.potential_conflict)
            slowest_same_dir = min(slowest_same_dir, temp_node.slowest_same_dir)
            min_dist_others_target = min(min_dist_others_target, temp_node.min_dist_others_target)
            dist_unusable_switch = temp_node.dist_unusable_switch + path_distance if dist_unusable_switch == np.inf else dist_unusable_switch

            malfunctioning_agent = max(malfunctioning_agent, temp_node.malfunctioning_agent)
            fastest_opposite_direction = max(fastest_opposite_direction, temp_node.fastest_opposite_direction)

            path_distance += temp_node.path_distace
            number_of_faster_agents_opposite_direction += temp_node.number_of_faster_agents_opposite_direction
            number_of_faster_agents_same_direction += temp_node.number_of_faster_agents_same_direction
            number_of_slower_agents_opposite_direction += temp_node.number_of_slower_agents_opposite_direction
            number_of_slower_agents_same_direction += temp_node.number_of_slower_agents_same_direction
            number_of_same_speed_agents_opposite_direction += temp_node.number_of_same_speed_agents_opposite_direction
            number_of_same_speed_agents_same_direction += temp_node.number_of_same_speed_agents_same_direction
            number_of_unusable_switches_on_path += temp_node.number_of_unusable_switches_on_path
            # number_of_usable_switches_on_path += temp_node.number_of_usable_switches_on_path
            dist_own_target += temp_node.dist_own_target
            sum_priorities_same_direction += temp_node.sum_priorities_same_direction
            sum_priorities_diff_direction += temp_node.sum_priorities_diff_direction

            if fastest_opposite_direction < temp_node.fastest_opposite_direction:
                fastest_opposite_direction = temp_node.fastest_opposite_direction
                dist_fastest_opposite_direction = temp_node.dist_fastest_opposite_direction

            if slowest_same_dir > temp_node.slowest_same_dir:
                slowest_same_dir = temp_node.slowest_same_dir
                dist_slowest_same_directon = temp_node.dist_slowest_same_directon

            deadlock = deadlock or temp_node.deadlock

            percentage_active_agents = temp_node.percentage_active_agents
            percentage_done_agents = temp_node.percentage_done_agents
            percentage_ready_to_depart_agents = temp_node.percentage_ready_to_depart_agents
            currently_on_switch = temp_node.currently_on_switch
            dist_usable_switch = temp_node.path_distace if dist_usable_switch == -1 else dist_usable_switch

            speeds_same_direction.extend(temp_node.statistical_data_info['speeds_same_direction'])
            speeds_diff_direction.extend(temp_node.statistical_data_info['speeds_diff_direction'])

            malfunctions_same_direction.extend(temp_node.statistical_data_info['malfunctions_same_direction'])
            malfunctions_diff_direction.extend(temp_node.statistical_data_info['malfunctions_diff_direction'])

            distances_to_target_same_direction.extend(
                temp_node.statistical_data_info['distances_to_target_same_direction'])
            distances_to_target_diff_direction.extend(
                temp_node.statistical_data_info['distances_to_target_diff_direction'])

            actions.extend(temp_node.actions)

        if len(speeds_same_direction) > 0:
            mean_agent_speed_same_direction = np.mean(speeds_same_direction)
            std_agent_speed_same_direction = np.std(speeds_same_direction)

        if len(speeds_diff_direction) > 0:
            mean_agent_speed_diff_direction = np.mean(speeds_diff_direction)
            std_agent_speed_diff_direction = np.std(speeds_diff_direction)

        if len(malfunctions_same_direction) > 0:
            mean_agent_malfunction_same_direction = np.mean(malfunctions_same_direction)
            std_agent_malfunction_same_direction = np.std(malfunctions_same_direction)

        if len(malfunctions_diff_direction) > 0:
            mean_agent_malfunction_diff_direction = np.mean(malfunctions_diff_direction)
            std_agent_malfunction_diff_direction = np.std(malfunctions_diff_direction)

        if len(distances_to_target_same_direction) > 0:
            mean_agent_distance_to_target_same_direction = np.mean(distances_to_target_same_direction)
            std_agent_distance_to_target_same_direction = np.mean(distances_to_target_same_direction)

        if len(distances_to_target_diff_direction) > 0:
            mean_agent_distance_to_target_diff_direction = np.mean(distances_to_target_diff_direction)
            std_agent_distance_to_target_diff_direction = np.mean(distances_to_target_diff_direction)

        return DQNShortestPath.Node(dist_opposite_direction=dist_opposite_direction,
                                    dist_same_direction=dist_same_direction,
                                    potential_conflict=potential_conflict,
                                    dist_fastest_opposite_direction=dist_fastest_opposite_direction,
                                    dist_slowest_same_directon=dist_slowest_same_directon,
                                    malfunctioning_agent=malfunctioning_agent,
                                    # Дали имаме некоја информација од ова?
                                    number_of_slower_agents_same_direction=number_of_slower_agents_same_direction,
                                    number_of_faster_agents_same_direction=number_of_faster_agents_same_direction,
                                    number_of_same_speed_agents_same_direction=number_of_same_speed_agents_same_direction,
                                    number_of_slower_agents_opposite_direction=number_of_slower_agents_opposite_direction,
                                    number_of_faster_agents_opposite_direction=number_of_faster_agents_opposite_direction,
                                    number_of_same_speed_agents_opposite_direction=number_of_same_speed_agents_opposite_direction,
                                    priority=priority,
                                    fastest_opposite_direction=fastest_opposite_direction,
                                    slowest_same_dir=slowest_same_dir,
                                    other_agent_wants_to_occupy_cell_in_path=other_agent_wants_to_occupy_cell_in_path,
                                    deadlock=deadlock,
                                    actions=actions,
                                    dist_own_target=dist_own_target,
                                    min_dist_others_target=min_dist_others_target,
                                    percentage_active_agents=percentage_active_agents,
                                    percentage_done_agents=percentage_done_agents,
                                    percentage_ready_to_depart_agents=percentage_ready_to_depart_agents,
                                    dist_usable_switch=dist_usable_switch,
                                    dist_unusable_switch=dist_unusable_switch,
                                    currently_on_switch=currently_on_switch,
                                    number_of_usable_switches_on_path=number_of_usable_switches_on_path,
                                    number_of_unusable_switches_on_path=number_of_unusable_switches_on_path,

                                    mean_agent_speed_same_direction=mean_agent_speed_same_direction,
                                    std_agent_speed_same_direction=std_agent_speed_same_direction,
                                    mean_agent_speed_diff_direction=mean_agent_speed_diff_direction,
                                    std_agent_speed_diff_direction=std_agent_speed_diff_direction,
                                    mean_agent_malfunction_same_direction=mean_agent_malfunction_same_direction,
                                    std_agent_malfunction_same_direction=std_agent_malfunction_same_direction,
                                    mean_agent_malfunction_diff_direction=mean_agent_malfunction_diff_direction,
                                    std_agent_malfunction_diff_direction=std_agent_malfunction_diff_direction,
                                    sum_priorities_same_direction=sum_priorities_same_direction,
                                    sum_priorities_diff_direction=sum_priorities_diff_direction,
                                    mean_agent_distance_to_target_same_direction=mean_agent_distance_to_target_same_direction,
                                    std_agent_distance_to_target_same_direction=std_agent_distance_to_target_same_direction,
                                    mean_agent_distance_to_target_diff_direction=mean_agent_distance_to_target_diff_direction,
                                    std_agent_distance_to_target_diff_direction=std_agent_distance_to_target_diff_direction,
                                    statistical_data_info={
                                                'speeds_same_direction': [],
                                                'speeds_diff_direction': [],
                                                'malfunctions_same_direction': [],
                                                'malfunctions_diff_direction': [],
                                                'distances_to_target_same_direction': [],
                                                'distances_to_target_diff_direction': [],
                                            },
                                    path_distace=path_distance
                                    )

    def _explore_path(self, handle, list_walking_elements, priority):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """
        agents = self.env.agents
        actions = []

        dist_unusable_switch = np.inf
        dist_opposite_direction = np.inf
        dist_same_direction = np.inf
        number_of_slower_agents_same_direction = 0
        number_of_faster_agents_same_direction = 0
        number_of_same_speed_agents_same_direction = 0
        number_of_slower_agents_opposite_direction = 0
        number_of_faster_agents_opposite_direction = 0
        number_of_same_speed_agents_opposite_direction = 0
        min_dist_other_target = np.inf
        potential_conflict = np.inf
        other_agent_wants_to_occupy_cell_in_path = False

        dist_fastest_opposite_direction = np.inf
        dist_slowest_same_directon = np.inf

        malfunctioning_agent = 0
        fastest_opposite_direction = 0
        slowest_same_dir = np.inf

        dist_usable_switch = -1
        currently_on_switch = 0
        number_of_usable_switches_on_path = 0
        number_of_unusable_switches_on_path = 0

        percentage_active_agents = -1
        percentage_done_agents = -1
        percentage_ready_to_depart_agents = -1

        mean_agent_speed_same_direction = 0
        std_agent_speed_same_direction = 0
        mean_agent_speed_diff_direction = 0
        std_agent_speed_diff_direction = 0
        mean_agent_malfunction_same_direction = 0
        std_agent_malfunction_same_direction = 0
        mean_agent_malfunction_diff_direction = 0
        std_agent_malfunction_diff_direction = 0
        sum_priorities_same_direction = 0
        sum_priorities_diff_direction = 0

        mean_agent_distance_to_target_same_direction = 0
        std_agent_distance_to_target_same_direction = 0
        mean_agent_distance_to_target_diff_direction = 0
        std_agent_distance_to_target_diff_direction = 0

        agent_on_position = {}
        my_speed = agents[handle].speed_data['speed']

        path_distance = 0

        if list_walking_elements is None:
            print("walking elements is none, ", agents[handle].status)
            node = DQNShortestPath.Node(dist_opposite_direction=dist_opposite_direction,
                                        dist_same_direction=dist_same_direction,
                                        potential_conflict=potential_conflict,
                                        dist_fastest_opposite_direction=dist_fastest_opposite_direction,
                                        dist_slowest_same_directon=dist_slowest_same_directon,
                                        malfunctioning_agent=malfunctioning_agent,
                                        number_of_slower_agents_same_direction=number_of_slower_agents_same_direction,
                                        number_of_faster_agents_same_direction=number_of_faster_agents_same_direction,
                                        number_of_same_speed_agents_same_direction=number_of_same_speed_agents_same_direction,
                                        number_of_slower_agents_opposite_direction=number_of_slower_agents_opposite_direction,
                                        number_of_faster_agents_opposite_direction=number_of_faster_agents_opposite_direction,
                                        number_of_same_speed_agents_opposite_direction=number_of_same_speed_agents_opposite_direction,
                                        priority=priority,
                                        fastest_opposite_direction=fastest_opposite_direction,
                                        slowest_same_dir=slowest_same_dir,
                                        other_agent_wants_to_occupy_cell_in_path=other_agent_wants_to_occupy_cell_in_path,
                                        deadlock=0,
                                        actions=actions,
                                        dist_own_target=0,
                                        min_dist_others_target=0,
                                        percentage_active_agents=percentage_active_agents,
                                        percentage_done_agents=percentage_done_agents,
                                        percentage_ready_to_depart_agents=percentage_ready_to_depart_agents,
                                        dist_usable_switch=dist_usable_switch,
                                        dist_unusable_switch=dist_unusable_switch,
                                        currently_on_switch=currently_on_switch,
                                        number_of_usable_switches_on_path=number_of_usable_switches_on_path,
                                        number_of_unusable_switches_on_path=number_of_unusable_switches_on_path,
                                        mean_agent_speed_same_direction=mean_agent_speed_same_direction,
                                        std_agent_speed_same_direction=std_agent_speed_same_direction,
                                        mean_agent_speed_diff_direction=mean_agent_speed_diff_direction,
                                        std_agent_speed_diff_direction=std_agent_speed_diff_direction,
                                        mean_agent_malfunction_same_direction=mean_agent_malfunction_same_direction,
                                        std_agent_malfunction_same_direction=std_agent_malfunction_same_direction,
                                        mean_agent_malfunction_diff_direction=mean_agent_malfunction_diff_direction,
                                        std_agent_malfunction_diff_direction=std_agent_malfunction_diff_direction,
                                        sum_priorities_same_direction=sum_priorities_same_direction,
                                        sum_priorities_diff_direction=sum_priorities_diff_direction,
                                        mean_agent_distance_to_target_same_direction=mean_agent_distance_to_target_same_direction,
                                        std_agent_distance_to_target_same_direction=std_agent_distance_to_target_same_direction,
                                        mean_agent_distance_to_target_diff_direction=mean_agent_distance_to_target_diff_direction,
                                        std_agent_distance_to_target_diff_direction=std_agent_distance_to_target_diff_direction,
                                        path_distance=path_distance)
            return node

        valid_actions = get_valid_move_actions_(agents[handle].direction,
                                                agents[handle].position if agents[handle].position else agents[
                                                    handle].initial_position, self.env.rail)
        if len(valid_actions) == 2:
            currently_on_switch = 1

        percentage_active_agents = 100 * len(
            [agent for agent in self.env.agents if agent.status == RailAgentStatus.ACTIVE]) / len(self.env.agents)
        percentage_done_agents = 100 * len([agent for agent in self.env.agents if agent.status in [RailAgentStatus.DONE,
                                                                                                   RailAgentStatus.DONE_REMOVED]]) / len(
            self.env.agents)
        percentage_ready_to_depart_agents = 100 * len(
            [agent for agent in self.env.agents if agent.status == RailAgentStatus.READY_TO_DEPART]) / len(
            self.env.agents)

        speeds_same_direction = []
        speeds_diff_direction = []

        malfunctions_same_direction = []
        malfunctions_diff_direction = []

        priorities_same_direction = []
        priorities_diff_direction = []

        distances_to_target_same_direction = []
        distances_to_target_diff_direction = []

        for idx, (action, position, direction, is_switch) in enumerate(list_walking_elements):
            if idx == 0:
                actions.append(action)
                continue

            valid_actions = get_valid_move_actions_(direction, position, self.env.rail)
            if len(valid_actions) == 2:
                number_of_usable_switches_on_path += 1
                if dist_usable_switch == -1:
                    dist_usable_switch = idx

            if position in self.location_has_target and position != self.env.agents[
                handle].target and min_dist_other_target != np.inf:
                min_dist_other_target = idx

            if position in self.location_has_agent:
                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                agent_on_path = self.location_has_agent[position]

                other_agent_priority = 0 if self.prioritites[self.location_has_agent[position].handle] == 0 and \
                                            self.priorities_second[self.prioritites[self.location_has_agent[position].handle]] == 0 else 1

                if self.location_has_agent_direction[position] == direction:
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < my_speed:
                        number_of_slower_agents_same_direction += 1
                    elif current_fractional_speed == my_speed:
                        number_of_same_speed_agents_same_direction += 1
                    else:
                        number_of_faster_agents_same_direction += 1

                    if current_fractional_speed < slowest_same_dir:
                        slowest_same_dir = current_fractional_speed
                        dist_slowest_same_directon = idx
                    if dist_same_direction == np.inf:
                        dist_same_direction = idx

                    speeds_same_direction.append(self.location_has_agent_speed[position])
                    malfunctions_same_direction.append(self.location_has_agent_malfunction[position])

                    priorities_same_direction.append(other_agent_priority)

                    distances_to_target_same_direction.append(
                        self.distance_map[agent_on_path.handle][agent_on_path.position[0]][agent_on_path.position[1]][
                            agent_on_path.direction])

                else:
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < my_speed:
                        number_of_slower_agents_opposite_direction += 1
                    elif current_fractional_speed == my_speed:
                        number_of_same_speed_agents_opposite_direction += 1
                    else:
                        number_of_faster_agents_opposite_direction += 1

                    if current_fractional_speed > fastest_opposite_direction:
                        fastest_opposite_direction = current_fractional_speed
                        dist_fastest_opposite_direction = idx

                    if dist_opposite_direction == np.inf:
                        dist_opposite_direction = idx

                    speeds_diff_direction.append(self.location_has_agent_speed[position])
                    malfunctions_diff_direction.append(self.location_has_agent_malfunction[position])

                    priorities_diff_direction.append(other_agent_priority)

                    distances_to_target_diff_direction.append(
                        self.distance_map[agent_on_path.handle][agent_on_path.position[0]][agent_on_path.position[1]][
                            agent_on_path.direction])

            # Register possible future conflict
            tot_dist = idx
            time_per_cell = np.reciprocal(my_speed)
            predicted_time = int(tot_dist * time_per_cell)
            cell_transitions = self.env.rail.get_transitions(*position, direction)

            # Register possible future conflict
            tot_dist = idx
            time_per_cell = np.reciprocal(my_speed)
            predicted_time = int(tot_dist * time_per_cell)

            cell_transitions = self.env.rail.get_transitions(*position, direction)

            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                    and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                    and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                    self.predicted_dir[post_step][ca])] == 1 \
                                    and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2

            num_transitions = np.count_nonzero(cell_transitions)

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions:
                number_of_unusable_switches_on_path += 1

                if dist_unusable_switch == np.inf:
                    dist_unusable_switch = idx

            actions.append(action)

        is_in_deadlock = False
        if dist_opposite_direction < np.inf and dist_opposite_direction <= dist_unusable_switch and dist_opposite_direction <= dist_usable_switch:
            is_in_deadlock = True

        # if dist_unusable_switch == np.inf:
        #     dist_unusable_switch = -1

        sum_priorities_same_direction = np.sum(priorities_same_direction)
        sum_priorities_diff_direction = np.sum(priorities_diff_direction)

        statistical_data_info = {
            'speeds_same_direction': speeds_same_direction,
            'speeds_diff_direction': speeds_diff_direction,
            'malfunctions_same_direction': malfunctions_same_direction,
            'malfunctions_diff_direction': malfunctions_diff_direction,
            'distances_to_target_same_direction': distances_to_target_same_direction,
            'distances_to_target_diff_direction': distances_to_target_diff_direction,
        }

        node = DQNShortestPath.Node(dist_opposite_direction=dist_opposite_direction,
                                    dist_same_direction=dist_same_direction,
                                    potential_conflict=potential_conflict,
                                    dist_fastest_opposite_direction=dist_fastest_opposite_direction,
                                    dist_slowest_same_directon=dist_slowest_same_directon,
                                    malfunctioning_agent=malfunctioning_agent,
                                    number_of_slower_agents_same_direction=number_of_slower_agents_same_direction,
                                    number_of_faster_agents_same_direction=number_of_faster_agents_same_direction,
                                    number_of_same_speed_agents_same_direction=number_of_same_speed_agents_same_direction,
                                    number_of_slower_agents_opposite_direction=number_of_slower_agents_opposite_direction,
                                    number_of_faster_agents_opposite_direction=number_of_faster_agents_opposite_direction,
                                    number_of_same_speed_agents_opposite_direction=number_of_same_speed_agents_opposite_direction,
                                    priority=priority,
                                    fastest_opposite_direction=fastest_opposite_direction,
                                    slowest_same_dir=slowest_same_dir,
                                    other_agent_wants_to_occupy_cell_in_path=other_agent_wants_to_occupy_cell_in_path,
                                    deadlock=is_in_deadlock,
                                    actions=actions,
                                    dist_own_target=idx + 1,  # IS OK??
                                    min_dist_others_target=min_dist_other_target,
                                    percentage_active_agents=percentage_active_agents,
                                    percentage_done_agents=percentage_done_agents,
                                    percentage_ready_to_depart_agents=percentage_ready_to_depart_agents,
                                    dist_usable_switch=dist_usable_switch,
                                    dist_unusable_switch=dist_unusable_switch,
                                    currently_on_switch=currently_on_switch,
                                    number_of_usable_switches_on_path=number_of_usable_switches_on_path,
                                    number_of_unusable_switches_on_path=number_of_unusable_switches_on_path,
                                    mean_agent_speed_same_direction=mean_agent_speed_same_direction,
                                    std_agent_speed_same_direction=std_agent_speed_same_direction,
                                    mean_agent_speed_diff_direction=mean_agent_speed_diff_direction,
                                    std_agent_speed_diff_direction=std_agent_speed_diff_direction,
                                    mean_agent_malfunction_same_direction=mean_agent_malfunction_same_direction,
                                    std_agent_malfunction_same_direction=std_agent_malfunction_same_direction,
                                    mean_agent_malfunction_diff_direction=mean_agent_malfunction_diff_direction,
                                    std_agent_malfunction_diff_direction=std_agent_malfunction_diff_direction,
                                    sum_priorities_same_direction=sum_priorities_same_direction,
                                    sum_priorities_diff_direction=sum_priorities_diff_direction,
                                    mean_agent_distance_to_target_same_direction=mean_agent_distance_to_target_same_direction,
                                    std_agent_distance_to_target_same_direction=std_agent_distance_to_target_same_direction,
                                    mean_agent_distance_to_target_diff_direction=mean_agent_distance_to_target_diff_direction,
                                    std_agent_distance_to_target_diff_direction=std_agent_distance_to_target_diff_direction,
                                    statistical_data_info=statistical_data_info,
                                    path_distace=idx + 1)

        return node

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)

    def print_node_features(node: Node):
        print("dist_opposite_direction=", node.dist_opposite_direction,
              ": potential_conflict=", node.potential_conflict,
              ", dist_fastest_opposite_direction=", node.dist_fastest_opposite_direction,
              ", dist_slowest_same_directon=", node.dist_slowest_same_directon,
              ", malfunctioning_agent=", node.malfunctioning_agent,
              ", number_of_slower_agents_same_direction=", node.number_of_slower_agents_same_direction,
              ", number_of_faster_agents_same_direction", node.number_of_faster_agents_same_direction,
              ", number_of_same_speed_agents_same_direction", node.number_of_same_speed_agents_same_direction,
              ", number_of_slower_agents_opposite_direction=", node.number_of_slower_agents_opposite_direction,
              ", number_of_faster_agents_opposite_direction", node.number_of_faster_agents_opposite_direction,
              ", number_of_same_speed_agents_opposite_direction", node.number_of_same_speed_agents_opposite_direction,
              ", priority=", node.priority,
              ", fastest_opposite_direction=", node.fastest_opposite_direction,
              ", slowest_same_dir=", node.slowest_same_dir,
              ", deadlock=", node.deadlock,
              "other_agent_wants_to_occupy_cell_in_path=", node.other_agent_wants_to_occupy_cell_in_path)

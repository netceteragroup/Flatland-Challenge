import numpy as np
import gym
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.envs.observations import GlobalObsForRailEnv
from typing import Tuple, Dict, List, Set, NamedTuple, Optional

from .segment_graph import Graph

from envs.flatland.observations import Observation, register_obs

# Features = NamedTuple('Features', [#("num_agents", int),
#                                    # ("width", int),
#                                    # ("height", int),
#                                    # ("max_num_cities", int),
#                                    # ("time_tick", int),
#                                    ("num_active_agents", int),
#                                    ("num_ready_agents", int),
#                                    ("deadlock_in_segment", int),
#                                    ("is_next_switch_usable", int),
#                                    ("shortest_path_length", int),
#                                    ("distance_left", int),
#                                    ("distance_forward", int),
#                                    ("distance_right", int),
#                                    ("number_agents_same_dir_on_shortest", int),
#                                    ("number_agents_opp_dir_on_shortest", int),
#                                    # ("alternative_path_dist", int),
#                                    # ("number_agents_same_dir_on_alternative", int),
#                                    # ("number_agents_opp_dir_on_alternative", int),
#                                    # ("number_of_switches_on_shortest_path", int),
#                                    ("potential_deadlock_left", int),
#                                    ("potential_deadlock_forward", int),
#                                    ("potential_deadlock_right", int),
#                                    # ("betweenness_switch_same_dir", float),
#                                    # ("betweenness_switch_opp_dir_avg", float),
#                                    # ("closeness_switch_same_dir", float),
#                                    # ("closeness_switch_opp_dir_avg", float),
#                                    # ("betweenness_shortest", float),
#                                    # ("betweenness_alternative", float),
#                                    # ("closeness_shortest", float),
#                                    # ("closeness_alternative", float),
#                                    ("is_on_switch", int),
#                                    ("dist_agent_same_dir", int),
#                                    ("dist_agent_opposite_dir", int),
#                                    ("dist_agent_same_dir_alternative", int),
#                                    ("dist_agent_opposite_dir_alternative", int),
#                                    ("dist_to_switch", int),
#                                    ("deadlock_on_segment_with_unusable_switches", int),
#                                    ("priority", int),
#                                    ("agent_status", int),
#                                    ])

Features = NamedTuple('Features', [
                                   ("agent_shortest_path_ind", int),
                                   ("priority", int),
                                   ("agent_status", int),
                                   ])

@register_obs("graphobs")
class GraphObservartion(Observation):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._builder = CustomGraphObservation()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=np.inf, shape=(len(Features._fields),))


class CustomGraphObservation(ObservationBuilder):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.graph = Graph(self.env)
        self.num_agents = len(self.env.agents)
        self.width = self.env.width
        self.height = self.env.height
        self.max_num_cities = (self.num_agents // 10) + 2

        self.time_tick = -1
        self.num_active_agents = len([agent for agent in self.env.agents if agent.status == RailAgentStatus.ACTIVE])
        self.num_ready_agents = len(
            [agent for agent in self.env.agents if agent.status == RailAgentStatus.READY_TO_DEPART])
        self.dist_next_switch = -1
        self.shortest_path_length = -1
        # for v, u, idx, data in self.graph.graph.edges(data=True, keys=True):
        #     print(f'{v},{u}    -  {data["segments"]}')
        #     print(edge[0], edge[1], edge[2])
        # for node in self.graph.graph.nodes():
        #     print(node, self.graph.graph.nodes[node]['action_dict'])
        # self.graph.draw_graph(self.graph.graph)

    def get(self, handle: int = 0, segment_deadlock=0):
        #
        #
        # self.graph.draw_graph(self.graph.graph)
        # self.graph.get_AgentInfo(handle)

        # print(handle, self.graph.segment_deadlock(handle))

        #
        #
        # self.graph.draw_graph(self.graph.graph)
        # self.graph.get_AgentInfo(handle)

        # print(handle, self.graph.segment_deadlock(handle))

        # print(handle, results)
        # print('===')

        # shortest_path_dist, number_agents_same_dir_on_shortest, \
        # number_agents_opp_dir_on_shortest, alternative_path_dist, number_agents_same_dir_on_alternative, \
        # number_agents_opp_dir_on_alternative, number_of_switches_on_shortest_path, betweenness_shortest, \
        # betweenness_alternative, closeness_shortest, closeness_alternative, dist_agent_same_dir, \
        # dist_agent_opposite_dir, dist_agent_same_dir_alternative, dist_agent_opposite_dir_alternative \
        #     = self.graph.compute_shortest_path(handle)
        # print(handle, dist_agent_same_dir, dist_agent_opposite_dir, dist_agent_same_dir_alternative,
        # dist_agent_opposite_dir_alternative)
        # print(handle, self.graph.compute_shortest_path(handle))

        # betweenness_switch_same_dir, betweenness_switch_opp_dir_avg = self.graph.get_centrality_for_next_node(handle,
        #                                                                                                       centrality="betweenness")
        # closeness_switch_same_dir, closeness_switch_opp_dir_avg = self.graph.get_centrality_for_next_node(handle,
                                                                                                          #centrality="closeness")
        # dist_to_switch = self.graph.dist_to_switch(handle)
        # deadlock_on_segment_with_unusable_switches = self.graph.check_if_unusable_switches_cause_deadlock(handle)
        priority = self.graph.priorities.get(handle, 0)
        agent_status = self.graph.get_agent_status(handle)

        # is_next_switch_usable = 1 if self.graph.check_if_next_switch_is_usable(handle) else 0
        # results, deadlocks = self.graph.all_paths_from_switch(handle)
        # potential_deadlock_left, potential_deadlock_forward, potential_deadlock_right = deadlocks
        # print(handle, results, deadlocks, is_next_switch_usable, shortest_path_dist, alternative_path_dist, priority)
        # deadlock_in_segment = 1 if self.graph.segment_deadlock(handle) else 0

        # is_on_switch = 1 if self.graph.is_on_switch(handle) else 0
        # prev_priority = self.graph.prev_priorities[handle] if handle in self.graph.prev_priorities.keys() else 0
        # print(handle, shortest_path_dist, deadlock_in_segment, results[0], results[1], results[2],
        #      potential_deadlock_left, potential_deadlock_forward, potential_deadlock_right, is_on_switch)
        # dist_shortest_alt = np.partition(results, 2)
        # shortest_path_dist = dist_shortest_alt[0]
        # alternative_path_dist = dist_shortest_alt[1]
        # if not is_on_switch or (is_on_switch and not is_next_switch_usable):
        #     results[1] = shortest_path_dist

        if agent_status == RailAgentStatus.DONE or agent_status == RailAgentStatus.DONE_REMOVED:
            min_path_ind = 3
        else:
            dist, visited, parent, start, best_path, end_node = self.graph.shortest_paths[handle]
            seg_with_station = [end_nodes for end_nodes in self.graph.agents[handle].EndEdges if
                                self.graph.agents[handle].CurrentNode in end_nodes]
            if len(seg_with_station) == 1:
                segment = self.graph.graph[seg_with_station[0][0]][seg_with_station[0][1]]["segment"]
                min_path_ind = self.graph.get_next_direction_from_given_direction_on_switch(segment[0][2],
                                                                                             segment[1][2])
            else:
                if end_node == 0 or end_node is None:
                    min_path_ind = 2
                else:
                    path = self.graph._construct_shortest_path(parent, end_node)
                    extended_path = self.graph._get_extended_shortest_path(self.graph.agents[handle], path)

                    min_path_ind = self.graph.get_next_direction_from_given_direction_on_switch(self.graph.agents[handle].Agent.direction,
                                                                                 extended_path[0][2])

        # print(handle, potential_deadlock_left, potential_deadlock_forward, potential_deadlock_right)
        # out = Features(#num_agents=self.num_agents,
        #                # width=self.width,
        #                # height=self.height,
        #                # max_num_cities=self.max_num_cities,
        #                # time_tick=self.time_tick,
        #                num_active_agents=self.num_active_agents,
        #                num_ready_agents=self.num_ready_agents,
        #                deadlock_in_segment=segment_deadlock,
        #                is_next_switch_usable=is_next_switch_usable,
        #                shortest_path_length=shortest_path_dist,
        #                distance_left=results[0],
        #                distance_forward=results[1],
        #                distance_right=results[2],
        #                number_agents_same_dir_on_shortest=number_agents_same_dir_on_shortest,
        #                number_agents_opp_dir_on_shortest=number_agents_opp_dir_on_shortest,
        #                # alternative_path_dist=alternative_path_dist,
        #                # number_agents_same_dir_on_alternative=number_agents_same_dir_on_alternative,
        #                # number_agents_opp_dir_on_alternative=number_agents_opp_dir_on_alternative,
        #                # number_of_switches_on_shortest_path=number_of_switches_on_shortest_path,
        #                potential_deadlock_left=potential_deadlock_left,
        #                potential_deadlock_forward=potential_deadlock_forward,
        #                potential_deadlock_right=potential_deadlock_right,
        #                # betweenness_switch_same_dir=betweenness_switch_same_dir,
        #                # betweenness_switch_opp_dir_avg=betweenness_switch_opp_dir_avg,
        #                # closeness_switch_same_dir=closeness_switch_same_dir,
        #                # closeness_switch_opp_dir_avg=closeness_switch_opp_dir_avg,
        #                # betweenness_shortest=betweenness_shortest,
        #                # betweenness_alternative=betweenness_alternative,
        #                # closeness_shortest=closeness_shortest,
        #                # closeness_alternative=closeness_alternative,
        #                is_on_switch=is_on_switch,
        #                dist_agent_same_dir=dist_agent_same_dir,
        #                dist_agent_opposite_dir=dist_agent_opposite_dir,
        #                dist_agent_same_dir_alternative=dist_agent_same_dir_alternative,
        #                dist_agent_opposite_dir_alternative=dist_agent_opposite_dir_alternative,
        #                dist_to_switch=dist_to_switch,
        #                deadlock_on_segment_with_unusable_switches=deadlock_on_segment_with_unusable_switches,
        #                priority=priority,
        #                agent_status=agent_status,
        #                )

        out = Features(
            agent_shortest_path_ind=min_path_ind,
            priority=priority,
            agent_status=agent_status,
        )

        # print(handle, potential_deadlock_left, potential_deadlock_forward, potential_deadlock_right, out[7], dist_to_switch)
        # print(handle, deadlock_on_segment_with_unusable_switches)
        # print(handle, deadlocks, betweenness_shortest, betweenness_alternative, closeness_shortest, closeness_alternative)
        # i = input()
        # print(out)
        out = np.array(out)
        # print(handle, results, self.graph.check_if_next_switch_is_usable(handle))
        # print('=============================================')
        return out

    def get_many(self, handles: Optional[List[int]] = None):
        self.num_active_agents = len([agent for agent in self.env.agents if agent.status == RailAgentStatus.ACTIVE])
        self.num_ready_agents = len(
            [agent for agent in self.env.agents if agent.status == RailAgentStatus.READY_TO_DEPART])
        self.time_tick += 1
        for handle in handles:
            self.graph.update_agent(handle)

        observations = {}
        deadlocks = []
        if handles is None:
            handles = []

        for h in handles:
            deadlocks.append(1 if self.graph.segment_deadlock(h) else 0)

        self.graph.compute_shortest_paths_for_all_agents()
        self.graph.calculate_priorities()

        for h in handles:
            observations[h] = self.get(h, deadlocks[h])

        return observations

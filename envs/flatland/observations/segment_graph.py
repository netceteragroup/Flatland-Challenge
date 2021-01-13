"""
    Author: Oliver Tanevski
    Created On: 9/4/2020
"""
from collections import deque
from heapq import *
from typing import NamedTuple, Tuple, List
import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.agent_utils import Agent, RailAgentStatus

AgentInfo = NamedTuple("AgentInfo", [("Agent", Agent),
                                     ("StartEdge", Tuple[Tuple, Tuple]),
                                     ("StartingNode", Tuple[int, int, int]),
                                     ("CurrentNode", Tuple[int, int, int]),
                                     ("NextNodes", List[Tuple[int, int, int]]),
                                     ("EndEdges", List[Tuple[Tuple]]),
                                     ("Deadlock", bool)])


class Graph:
    agents = {}
    graph_global = None

    def __init__(self, env):
        self.env = env
        self.agents = {}
        self.segment_dict = {}
        self.graph = self._create_graph()
        Graph.graph_global = self.graph
        Graph.agents = self.agents
        self.shortest_paths = {}
        self.priorities = {}
        self.prev_priorities = {}

    def is_on_switch(self, handle):
        if (*self._get_virtual_position(self.agents[handle].Agent),
            self.agents[handle].Agent.direction) in self.graph.nodes():
            return True
        return False

    @staticmethod
    def get_virtual_position(handle):
        agent_virtual_position = None
        agent = Graph.agents[handle].Agent
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            agent_virtual_position = agent.target

        return agent_virtual_position

    def _create_graph(self):
        G = self._create_big_graph()
        # self.draw_graph(G)
        G = self._reduce_big_graph(G)
        # self.draw_graph(G)
        self._init_segment_dict(G)
        # print(self.segment_dict)
        self._init_agents(G)

        G = self._init_centrality_metrics(G)

        return G

    def _init_centrality_metrics(self, G):
        bc = nx.betweenness_centrality(G, weight='weight')
        cc = nx.closeness_centrality(G, distance='weight')

        nx.set_node_attributes(G, bc, "betweenness")
        nx.set_node_attributes(G, cc, "closeness")

        return G

    def _create_big_graph(self) -> nx.DiGraph:
        """
        Create a directed graph between each grid that has a rail in it.
        The node in the graph is saved in format (i, j, direction).
        An edge between (i1, j1, d1) and (i2, j2, d2) can be interpreted like:
        If agent has position (i1, j1) and its orientation is d1, he can go to (i2, j2) and his orientation will be d2

        :return: networkx Digraph of grid cells
        """
        G = nx.DiGraph()
        for i in range(self.env.height):
            for j in range(self.env.width):
                coord_transition = self.env.rail.get_full_transitions(i, j)
                # if rail detected
                if coord_transition != 0:
                    for direction in [0, 1, 2, 3]:

                        cell_transition = [i for i, v in enumerate(self.env.rail.get_transitions(*(i, j), direction)) if
                                           v == 1]
                        for cc in cell_transition:
                            dx, dy = self._get_coords(cc)
                            # if it is a dead end
                            if coord_transition & 0b0010000000000000 > 0:
                                G.add_edge((i, j, direction), (i, j, cc))
                                G.add_edge((i, j, cc), (i + dx, j + dy, cc))
                            else:
                                G.add_edge((i, j, direction), (i + dx, j + dy, cc))

        return G

    def _reduce_big_graph(self, G) -> nx.DiGraph:
        """
        Creates a reduced version of the directed graph of gird cells where nodes are switches with certain orientation.

        :param G: networkx Digraph of grid cells
        :return: networkx Digraph of segments where the nodes are the switches (with certain orientation)
        """
        new_G = nx.DiGraph()
        switches = [n for n in G.nodes() if len(list(G.successors(n))) >= 2]
        switches_unique = set((u, v) for u, v, _ in switches)

        # print(switches)
        # print(switches_unique)

        for switch in switches_unique:
            for dir in [0, 1, 2, 3]:
                # check in what direction the switch can be accessed
                if (switch[0], switch[1], dir) in G:
                    succs = list(G.successors((switch[0], switch[1], dir)))
                    for successor in succs:
                        l = [((switch[0], switch[1], dir))]
                        if (successor[0], successor[1]) in switches_unique:
                            l.append(successor)

                            new_G.add_edge(((switch[0], switch[1], dir)), successor, weight=len(l) - 1, orig_weight=len(l) - 1, segment=l)
                            continue
                        else:
                            l.append(successor)
                            dq = deque(list(G.successors(successor)))

                            while dq:
                                elem = dq.popleft()
                                if (elem[0], elem[1]) in switches_unique:
                                    l.append(elem)
                                    new_G.add_edge(((switch[0], switch[1], dir)), elem, weight=len(l) - 1, orig_weight=len(l) - 1, segment=l)
                                    break
                                else:
                                    l.append(elem)
                                    dq.append(list(G.successors(elem))[0])
        return new_G

    def _init_segment_dict(self, graph: nx.DiGraph):
        """
        Initializes a segment dictionary where the key is the segment (without directions)
        and values are edges that contain that segment and also information if there is deadlock

        :param graph: networkx Directed graph of segments
        :return: None
        """
        for e1, e2, data in graph.edges(data=True):
            segment_grid = frozenset((cx, cy) for cx, cy, _ in data["segment"])
            if segment_grid not in self.segment_dict:
                self.segment_dict[segment_grid] = {"edges": [(e1, e2)], "deadlock": False,
                                                   "list_malfunction_duration": [0] * len(self.env.agents),
                                                   "agents_on_segment": set()}
            else:
                self.segment_dict[segment_grid]["edges"].append((e1, e2))

    def _init_agents(self, graph: nx.DiGraph):
        """
        Initializes agents with certain info (check AgentInfo)

        :param graph: networkx Directed graph of segments
        :return: None
        """
        for agent in self.env.agents:
            foundStart = False
            EndEdges = []
            for e1, e2, data in graph.edges(data=True):
                # print(e1, e2, data, agent.position, agent)
                segment_grid = [(cx, cy) for cx, cy, _ in data["segment"]]
                if not foundStart:
                    agent_idx = [i for i, grid_cell in enumerate(data["segment"]) if
                                 grid_cell[0] == agent.initial_position[0] and
                                 grid_cell[1] == agent.initial_position[1] and grid_cell[2] == agent.direction]
                    if len(agent_idx) > 0:
                        foundStart = True
                        StartEdge = (e1, e2)
                        StartingNode = e1
                        CurrentNode = e1
                        NextNodes = [e2]
                        hashed_cells = frozenset(segment_grid)
                        self.segment_dict[hashed_cells]["agents_on_segment"].add(agent.handle)

                if agent.target in segment_grid:
                    EndEdges.append((e1, e2))
            self.agents[agent.handle] = AgentInfo(Agent=agent, StartEdge=StartEdge, StartingNode=StartingNode,
                                                  CurrentNode=CurrentNode, NextNodes=NextNodes, EndEdges=EndEdges,
                                                  Deadlock=False)

    def update_agent(self, handle: int):
        """
        Updates CurrentNode and NextNodes for the given agent

        :param handle: ID of the agent
        :return: None
        """

        agent = self.agents[handle]

        for next_node in agent.NextNodes:
            agent_position = self._get_virtual_position(agent.Agent)
            if next_node[0] == agent_position[0] and next_node[1] == \
                    agent_position[1] \
                    and next_node[2] == agent.Agent.direction:
                NextNodes = list(self.graph.successors(next_node))

                outer_hashed_cells = frozenset(
                    (cx, cy) for cx, cy, _ in self.graph[agent.CurrentNode][next_node]["segment"])
                self.segment_dict[outer_hashed_cells]["agents_on_segment"].remove(handle)

                for nn in NextNodes:
                    hashed_cells = frozenset((cx, cy) for cx, cy, _ in self.graph[next_node][nn]["segment"])
                    self.segment_dict[hashed_cells]["agents_on_segment"].add(handle)

                self.agents[handle] = self.agents[handle]._replace(NextNodes=NextNodes)
                self.agents[handle] = self.agents[handle]._replace(CurrentNode=next_node)
                break
            elif len(agent.NextNodes) > 1 and (agent.CurrentNode[0], agent.CurrentNode[1]) != agent.Agent.position:
                edge = self.graph[agent.CurrentNode][next_node]
                # print((agent.Agent.position, agent.Agent.direction))
                if (*agent_position, agent.Agent.direction) in edge["segment"]:
                    for nn in [node for node in agent.NextNodes if next_node != node]:
                        hashed_cells = frozenset((cx, cy) for cx, cy, _ in self.graph[agent.CurrentNode][nn]["segment"])
                        self.segment_dict[hashed_cells]["agents_on_segment"].remove(handle)

                    self.agents[handle] = self.agents[handle]._replace(NextNodes=[next_node])
                    break

        # update malfunction
        for next_node in agent.NextNodes:
            edge = self.graph[agent.CurrentNode][next_node]["segment"]
            hashed_segment = frozenset((cx, cy) for cx, cy, _ in edge)
            self.segment_dict[hashed_segment]["list_malfunction_duration"][handle] = agent.Agent.malfunction_data[
                "malfunction"]
            if self.graph[agent.CurrentNode][next_node]["weight"] < 5000:
                self.graph[agent.CurrentNode][next_node]["weight"] = self.graph[agent.CurrentNode][next_node]["orig_weight"] + max(self.segment_dict[hashed_segment]["list_malfunction_duration"])

        if agent.Agent.status == RailAgentStatus.DONE_REMOVED:
            edge = self.graph[agent.CurrentNode][agent.NextNodes[0]]["segment"]
            hashed_segment = frozenset((cx, cy) for cx, cy, _ in edge)
            agents_on_segment = self.segment_dict[hashed_segment]["agents_on_segment"]
            if handle in agents_on_segment:
                self.segment_dict[hashed_segment]["agents_on_segment"].remove(handle)

    def dist_to_switch(self, handle):
        agent = self.agents[handle]
        dist = 0
        if len(agent.NextNodes) > 1:
            return dist
        else:
            pos_agent = self._get_virtual_position(agent.Agent)
            if pos_agent == (agent.CurrentNode[0], agent.CurrentNode[1]):
                return dist

            agent_segment = self.graph[agent.CurrentNode][agent.NextNodes[0]]["segment"]
            agent_pos_idx_segment = [i for i, p in enumerate(agent_segment) if (p[0], p[1]) == pos_agent][0]
            dist = (len(agent_segment) - 1) - agent_pos_idx_segment

        return dist

    def station_on_segment(self, handle: int):
        agent = self.agents[handle]
        if len(agent.NextNodes) == 2:
            return 0

        else:
            if (agent.CurrentNode, agent.NextNodes[0]) in agent.EndEdges:
                return 1
            else:
                return 0

    def get_agent_status(self, handle):
        return self.agents[handle].Agent.status

    def _get_virtual_position(self, agent: Agent) -> Tuple[int]:
        """
        :param agent: the agent itself
        :return: the position of the agent depending on the status.
        """

        agent_virtual_position = None

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            agent_virtual_position = agent.target

        return agent_virtual_position

    def get_next_direction_from_given_direction_on_switch(self, direction: int, next_direction: int) -> int:
        """
        :param direction: the current direction of an agent
        :param next_direction: the next direction of an agent
        :return: the next available direction for an agent
        """
        # forward
        if direction == next_direction:
            return 1
        # right
        elif ((direction + 1) % 4) == next_direction:
            return 2
        # left
        elif ((direction - 1) % 4) == next_direction:
            return 0

    def get_centrality_for_next_node(self, handle, centrality='betweenness'):
        """
        :param handle: ID of the agent
        :param centrality: the centrality name
        :return: centrality score for the switch from my direction, average centrality score for the switch from all other directions
        """
        agent = self.agents[handle]
        if len(agent.NextNodes) > 1:
            switch = agent.CurrentNode
        else:
            switch = agent.NextNodes[0]

        centrality_switch_same_dir = self.graph.nodes[switch][centrality]
        centrality_switch_dirs_value = []
        for dir in range(4):
            if (switch[0], switch[1], dir) in self.graph and dir != switch[2]:
                centrality_switch_dirs_value.append(self.graph.nodes[(switch[0], switch[1], dir)][centrality])

        return centrality_switch_same_dir, np.mean(centrality_switch_dirs_value)

    def check_if_next_switch_is_usable(self, handle: int) -> bool:
        """
        :param handle: ID of the agent
        :return: boolean value whether the next switch is usable for the agent
        """
        agent = self.agents[handle]
        if len(agent.NextNodes) == 1:
            return True if len(list(self.graph.successors(agent.NextNodes[0]))) > 1 else False
        # if we are currently on switch
        else:
            return True

    def _check_all_next_nodes_for_deadlock(self, next_nodes, segment):
        for node in next_nodes:
            if node[0] == segment[0] and node[1] == segment[1]:
                return True
        return False

    def check_whether_deadlock_is_present(self, handle, long_segment):
        if len(long_segment) == 0:
            return 0

        for agent_id in self.agents:
            other_agent = self.agents[agent_id]
            if handle != agent_id and other_agent.Agent.status == RailAgentStatus.ACTIVE:
                pos_other_agent = self._get_virtual_position(other_agent.Agent)
                idx_other_agent_list = [i for i, p in enumerate(long_segment) if (p[0], p[1]) == pos_other_agent]
                if len(idx_other_agent_list) > 0:
                    idx_other_agent = idx_other_agent_list[0]
                    if other_agent.Agent.direction != long_segment[idx_other_agent][2] or other_agent.Deadlock:
                        return 1

        return 0

    def _create_long_segment(self, next_n):
        long_segment = []
        while True:
            nn = list(self.graph.successors(next_n))
            if len(nn) > 1:
                long_segment.append(next_n)
                break

            long_segment.extend(self.graph[next_n][nn[0]]['segment'][:-1])
            next_n = nn[0]

        return long_segment

    def _potential_deadlock_after_first_usable(self, handle):
        agent = self.agents[handle]
        first_usable = None if len(agent.NextNodes) == 1 else agent.CurrentNode
        next_n = agent.NextNodes[0]
        while first_usable is None:
            nn = list(self.graph.successors(next_n))
            if len(nn) > 1:
                first_usable = next_n
                break
            next_n = nn[0]
        deadlock_for_nn = [-1, -1, -1]
        dir = first_usable[2]
        for nn in list(self.graph.successors(first_usable)):
            next_dir = self.graph[first_usable][nn]['segment'][1][2]
            ind = self.get_next_direction_from_given_direction_on_switch(dir, next_dir)
            long_segment = []
            long_segment.extend(self.graph[first_usable][nn]['segment'][:-1])
            long_segment.extend(self._create_long_segment(nn))
            deadlock_for_nn[ind] = self.check_whether_deadlock_is_present(handle, long_segment)

        return deadlock_for_nn

    def check_if_unusable_switches_cause_deadlock(self, handle):
        agent = self.agents[handle]
        if len(agent.NextNodes) > 1:
            return 0

        next_n = agent.NextNodes[0]
        long_segment = self._create_long_segment(next_n)

        return self.check_whether_deadlock_is_present(handle, long_segment)

    def _get_deadlock_info(self, segment, handle) -> int:
        """
        checks whether there exists a potential deadlock
        :param segment: the segment where the agent wants to turn
        :param handle: ID of the agent
        :return: boolean 1 or 0 depending on wether there is a potential deadlock on the segment or not
        """
        for agent_id in self.agents:
            if agent_id == handle:
                continue
            agent = self.agents[agent_id]

            curr_position = self._get_virtual_position(agent.Agent)
            for i, pos in enumerate(segment):
                p = (pos[0], pos[1])
                if curr_position == p and (agent.Agent.direction != pos[2] or agent.Deadlock) \
                        and self.env.distance_map.get()[(agent.Agent.handle, *self._get_virtual_position(agent.Agent),
                                                         agent.Agent.direction)] > len(segment) - i:
                    return 1
        return 0

    def all_paths_from_switch(self, handle: int) -> (np.ndarray, np.ndarray):
        """
        This function returns a list containing the shortest distances for every possible
        action when on a switch. If the agent is not a switch, it returns the shortest path to the target.

        The values returned can be:
            - some small value: meaning there is a path to target
            - 5000: meaning we can't go to that direction if we turned (left, right, forward) or there is deadlock
            - 5001 and above : meaning that target can't be reached from this position (there is no path)

        :param handle: agent id
        :return: List["left", "forward", "right"]

        """
        agent = self.agents[handle]
        distance_map = self.env.distance_map.get()
        virtual_position = self._get_virtual_position(agent.Agent)

        # if the agent is DONE but his station is on a switch...
        if agent.Agent.status == RailAgentStatus.DONE_REMOVED:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        elif self.station_on_segment(handle) == 1:
            return np.array([distance_map[(handle, *virtual_position, agent.Agent.direction)]]*3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        results = np.full(3, 5000., dtype=np.float32)
        potential_deadlock_after_first_usable = self._potential_deadlock_after_first_usable(handle)

        if distance_map[(handle, *virtual_position, agent.Agent.direction)] == np.inf:
            return np.full(3, 5001, dtype=np.float32), potential_deadlock_after_first_usable

        first_usable = None if len(agent.NextNodes) == 1 else agent.CurrentNode
        next_n = agent.NextNodes[0]
        while first_usable is None:
            nn = list(self.graph.successors(next_n))
            if len(nn) > 1:
                first_usable = next_n
                break
            next_n = nn[0]

        for nn in list(self.graph.successors(first_usable)):
                agent_segment = self.graph[first_usable][nn]["segment"]
                hashed_grid_cells = frozenset((cx, cy) for cx, cy, _ in agent_segment)
                deadlock_on_segment = self.segment_dict[hashed_grid_cells]["deadlock"]
                malfunction_on_segment = max(self.segment_dict[hashed_grid_cells]["list_malfunction_duration"])
                next_dir = agent_segment[1][2]
                next_pos = (agent_segment[1][0], agent_segment[1][1])
                ind = self.get_next_direction_from_given_direction_on_switch(agent_segment[0][2], next_dir)

                if deadlock_on_segment:
                    results[ind] = 5000.
                else:
                    # we add the 5000 in case there is no path
                    results[ind] = min(distance_map[(handle, *next_pos, next_dir)], 5000.) + 1.
                    if results[ind] < 5000:
                        # if there is malfunction
                        if malfunction_on_segment > 0:
                            other_agent_idx = self.segment_dict[hashed_grid_cells]["list_malfunction_duration"].index(
                                malfunction_on_segment)
                            other_agent_pos = self._get_virtual_position(self.agents[other_agent_idx].Agent)
                            other_agent_idx_pos = [i for i, p in enumerate(agent_segment) if
                                                   (p[0], p[1]) == other_agent_pos][0]

                            dist_to_other_agent = abs(distance_map[(handle, *virtual_position, agent.Agent.direction)] - distance_map[(handle, *(agent_segment[0][0], agent_segment[0][1]), agent_segment[0][2])]) + other_agent_idx_pos

                            if dist_to_other_agent < malfunction_on_segment:
                                results[ind] += abs(malfunction_on_segment - dist_to_other_agent)

                        results[ind] += abs(distance_map[(handle, *virtual_position, agent.Agent.direction)] - distance_map[(handle, *(agent_segment[0][0], agent_segment[0][1]), agent_segment[0][2])])

        for i, d in enumerate(potential_deadlock_after_first_usable):
            if d == -1 or d == 1:
                results[i] = 5000

        return results, potential_deadlock_after_first_usable

    def _switch_dir(self, direction: int) -> int:
        """
        :param direction: the direction of an agent
        :return: the opposite direction to the direction given as argument
        """
        return (direction + 2) % 4

    def _num_agents_on_path(self, handle: int, path: List[Tuple[int]]) -> (int, int):
        """
        :param handle: ID of the agent
        :param path: the path the agent is supposed to take
        :return: the number of agents on the path that have the same or opposite direction to the agent
        """
        number_same_dir = 0
        number_opposite_dir = 0
        dist_agent_opposite_dir = 1 << 20
        dist_agent_same_dir = 1 << 20

        for other_agent_id in self.agents:
            if other_agent_id == handle or self.agents[other_agent_id].Agent.status == RailAgentStatus.DONE or \
                    self.agents[other_agent_id].Agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            other_agent = self.agents[other_agent_id]
            other_agent_pos = self._get_virtual_position(other_agent.Agent)
            other_agent_path_pos = [i for i, s in enumerate(path) if other_agent_pos == (s[0], s[1])]
            if len(other_agent_path_pos) > 0:
                cell = path[other_agent_path_pos[0]]
                if other_agent.Agent.direction == cell[2]:
                    number_same_dir += 1
                    dist_agent_same_dir = min(dist_agent_same_dir, other_agent_path_pos[0])
                elif other_agent.Agent.direction != cell[2]:
                    number_opposite_dir += 1
                    dist_agent_opposite_dir = min(dist_agent_opposite_dir, other_agent_path_pos[0])

        return number_same_dir, number_opposite_dir, dist_agent_same_dir + 1 if dist_agent_same_dir != 1 << 20 else -1, \
               dist_agent_opposite_dir + 1 if dist_agent_opposite_dir != 1 << 20 else -1

    def _init_dijkstra_for_agent(self, agent: Agent, source=None) -> ({}, {}, {}, Tuple[int]):
        """
        Initializes the needed data structures in order to run the dijkstra algorithm for the agent.
        :param agent: ID of the agent
        :param source: the starting position of the agent (this is needed only for the alternative path)
        :return: the distance dictionary, the visited dictionary, the parent dictionary and the starting node for the agent.
        """
        dist = dict()
        visited = dict()
        parent = dict()
        for node in self.graph.nodes:
            dist[node] = np.inf
            visited[node] = False
            parent[node] = None

        starting_node = None
        next_nodes = None
        if agent is None:
            current_node = source
        else:
            next_nodes = agent.NextNodes
            current_node = agent.CurrentNode

        if next_nodes is not None and len(next_nodes) == 1:
            start_segment = self.graph[current_node][next_nodes[0]]['segment']
            for i, cell in enumerate(start_segment):
                x = cell[0]
                y = cell[1]
                position = self._get_virtual_position(agent.Agent)
                if x == position[0] and y == position[1]:
                    dist[next_nodes[0]] = len(start_segment) - i - 1
                    starting_node = next_nodes[0]
                    break
        else:
            dist[current_node] = 0
            starting_node = current_node
        return dist, visited, parent, starting_node

    def _run_dijkstra(self, dist, visited, parent, start):
        """
        This function runs the dijkstra algorithm, given all the data structures that are needed.
        :param dist: the distance dictionary
        :param visited: the visited dictionary
        :param parent: the parent dictionary
        :param start: the starting node
        :return: the updated distance dictionary containing shortest paths from start to all other nodes and the updated
        parent dictionary that contains the parent of every node.
        """
        q = [(dist[start], start)]
        while q:
            (_, node) = heappop(q)
            if not visited[node]:
                visited[node] = True

                for nn in self.graph.neighbors(node):
                    weight = self.graph[node][nn]['weight']
                    if dist[nn] > dist[node] + weight:
                        dist[nn] = dist[node] + weight
                        parent[nn] = node
                        heappush(q, (dist[nn], nn))

        return dist, parent

    def _pick_end_node_of_shortest_path(self, dist: {}, agent: Agent) -> (int, Tuple[int]):
        """
        This function decides which of the two end nodes of an agent is the one he needs to go to in order to follow the
        shortest path
        :param dist: the distance dictionary
        :param agent: the agent itself
        :return: the shortest path between the agent and the target and the end node.
        """
        to_add_left = 0
        to_add_right = 0
        u, v = agent.EndEdges[0]
        seg = self.graph[u][v]['segment']
        for i, tup in enumerate(seg):
            x = tup[0]
            y = tup[1]
            if x == agent.Agent.target[0] and y == agent.Agent.target[1]:
                to_add_left = i
                to_add_right = len(seg) - i - 1

        best = np.inf
        end_node = None
        for x in agent.EndEdges:
            if (*self._get_virtual_position(agent.Agent), agent.Agent.direction) \
                    in [(xx, yy, d) for xx, yy, d
                        in self.graph[x[0]][x[1]]['segment']]:
                best = self.env.distance_map.get()[
                    (agent.Agent.handle, *self._get_virtual_position(agent.Agent), agent.Agent.direction)]
                return best, 0

            if dist[x[0]] + to_add_left < dist[x[1]] + to_add_right \
                    and best > dist[x[0]] + to_add_left:
                end_node = x[0]
                best = dist[x[0]] + to_add_left
            elif dist[x[0]] + to_add_left >= dist[x[1]] + to_add_right \
                    and best > dist[x[1]] + to_add_right:
                end_node = x[1]
                best = dist[x[1]] + to_add_right

        return best, end_node

    def _fill_path_list_if_on_last_segment(self, agent: Agent) -> List[Tuple[int]]:
        """
        This function constructs the agents path if he is on the segment that contains its target.
        :param agent: the agent itself
        :return: the path to the agents target
        """
        path_if_on_same_edge_to_target = []
        end_node = None
        for end in agent.EndEdges:
            if end[0] == agent.CurrentNode:
                end_node = end[1]
                break
        end_segment = self.graph[agent.CurrentNode][end_node]['segment']
        flag = False
        for cell in end_segment:
            if flag:
                path_if_on_same_edge_to_target.append(cell)
            elif (cell[0], cell[1]) == self._get_virtual_position(agent.Agent):
                flag = True
            if (cell[0], cell[1]) == agent.Agent.target:
                break
        return path_if_on_same_edge_to_target

    def _construct_shortest_path(self, parent: {}, end_node: Tuple[int]) -> List[Tuple[int]]:
        """
        Given the parent dictionary and the end node, this function constructs the shortest path to the agents target.
        :param parent: the parent dictionary
        :param end_node: the end node
        :return: a list containing the nodes that make the shortest path to the agents target
        """
        path = []
        while parent[end_node] is not None:
            path.insert(0, end_node)
            end_node = parent[end_node]
        path.insert(0, end_node)
        return path

    def _get_extended_shortest_path(self, agent: Agent, path: List[Tuple[int]]) -> List[Tuple[int]]:
        """
        :param agent: the agent itself
        :param path: a given path to one of its end nodes
        :return: an extended version of the same path which also contains the cells between the switches
        """
        new_path = []
        while len(path) > 0 and path[0] not in self.graph.successors(agent.CurrentNode):
            path = path[1:]

        if len(path) == 0:
            return []

        if agent.CurrentNode != path[0]:
            # print(agent.CurrentNode, path[0], 'ednakvi li ste')
            flag = False
            first_seg = self.graph[agent.CurrentNode][path[0]]['segment']
            for cell in first_seg:
                if flag:
                    new_path.append(cell)
                elif (cell[0], cell[1]) == self._get_virtual_position(agent.Agent):
                    flag = True

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            segment = self.graph[u][v]['segment'][1:]
            new_path.extend(segment)

        end_edge = None

        for end in agent.EndEdges:
            if end[0] == new_path[-1]:
                end_edge = end
                break

        segment = self.graph[end_edge[0]][end_edge[1]]['segment'][1:]
        for cell in segment:
            new_path.append(cell)
            if (cell[0], cell[1]) == agent.Agent.target:
                break
        return new_path

    def _generate_alternative_path(self, path: List[Tuple[int]], dist: {}, agent: Agent) -> (List[Tuple[int]], int):
        """
        :param path: the shortest path of the agent
        :param dist: the distance dictionary
        :param agent: the agent itself
        :return: an alternative path to the shortest path and its length
        """
        remember_branch_node = None
        alternative_path = []
        alternative_path_length = 1 << 30
        flag = False
        for node in path:
            neighbors = list(self.graph.neighbors(node))
            if flag:
                break
            if len(neighbors) > 1:
                for neighbor in neighbors:
                    if neighbor not in path:
                        dist2, visited, parent, start = self._init_dijkstra_for_agent(agent=None, source=neighbor)
                        dist2, parent = self._run_dijkstra(dist2, visited, parent, start)
                        second_best, end_node = self._pick_end_node_of_shortest_path(dist2, agent)
                        if end_node is not None:
                            alternative_path = self._construct_shortest_path(parent, end_node)
                            alternative_path_length = second_best + dist[neighbor]
                            flag = True
                            remember_branch_node = node
                            break
        if alternative_path_length != 1 << 30:
            for i in range(len(path)):
                if remember_branch_node != path[i]:
                    alternative_path.insert(i, path[i])
                else:
                    alternative_path.insert(i, path[i])
                    break

        for x in agent.EndEdges:
            if len(alternative_path) <= 1 or x == (alternative_path[-2], alternative_path[-1]):
                alternative_path = []
                alternative_path_length = 1 << 30
                break

        return alternative_path, alternative_path_length

    def compute_shortest_paths_for_all_agents(self):
        for a in self.env.agents:
            handle = a.handle
            agent = self.agents[handle]
            if agent.Agent.status == RailAgentStatus.DONE or agent.Agent.status == RailAgentStatus.DONE_REMOVED:
                self.shortest_paths[handle] = None
                continue

            elif handle not in self.shortest_paths or (agent.Agent.status != RailAgentStatus.READY_TO_DEPART and (agent.Agent.old_position is not None and agent.Agent.old_position != agent.Agent.position)):
                dist, visited, parent, start = self._init_dijkstra_for_agent(agent)
                dist, parent = self._run_dijkstra(dist, visited, parent, start)

                best_path, end_node = self._pick_end_node_of_shortest_path(dist, agent)

                self.shortest_paths[handle] = (dist, visited, parent, start, best_path, end_node)

    def compute_shortest_path(self, handle):
        """
        :param handle: ID of the agent
        :return: shortest path length, the number of agents on the shortest path facing in the same direction as the
        agent, the number of agents on the shortest path facing the opposite direction to the agents direction,
        the alternative path length, the number of agents on the alternative facing in the same direction as the
        agent, the number of agents on the alternative path facing the opposite direction to the agents direction
        number of switches in shortest path, average betweenness centrality for switches in shortest path,
        average betweenness centrality for switches in alternative path, average closeness centrality for switches in shortest path,
        average closeness centrality for switches in alternative path
        """

        agent = self.agents[handle]
        if agent.Agent.status == RailAgentStatus.DONE or agent.Agent.status == RailAgentStatus.DONE_REMOVED:
            return 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

        path_if_on_same_edge_to_target = []

        dist, visited, parent, start, best_path, end_node = self.shortest_paths[handle]
        if end_node == 0:
            path_if_on_same_edge_to_target = self._fill_path_list_if_on_last_segment(agent)

        if end_node is None:
            return -1, -1, -1, -1, -1, -1, 0, -1., -1., -1., -1., -1, -1, -1, -1

        if len(path_if_on_same_edge_to_target) > 0:
            n_same_dir, n_opp_dir, dist_agent_same_dir, dist_agent_opposite_dir = \
                self._num_agents_on_path(handle, path_if_on_same_edge_to_target)
            return best_path, n_same_dir, n_opp_dir, -1, -1, -1, 0, 0., 0., -1., -1., dist_agent_same_dir, \
                   dist_agent_opposite_dir, -1, -1

        shortest_path = self._construct_shortest_path(parent, end_node)
        alternative_path, alternative_path_length = self._generate_alternative_path(shortest_path, dist, agent)
        number_of_switches_on_shortest = len(shortest_path)

        extended_shortest_path = self._get_extended_shortest_path(agent, shortest_path)
        number_of_agents_on_path_same_dir, number_of_agents_on_path_opposite_dir, dist_agent_same_dir, \
        dist_agent_opposite_dir = self._num_agents_on_path(handle, extended_shortest_path)

        if alternative_path_length == 1 << 30:
            betweenness_shortest_avg = np.mean([self.graph.nodes[n]['betweenness'] for n in shortest_path])
            closeness_shortest_avg = np.mean([self.graph.nodes[n]['closeness'] for n in shortest_path])
            return best_path, number_of_agents_on_path_same_dir, number_of_agents_on_path_opposite_dir, -1, -1, -1, \
                   number_of_switches_on_shortest, betweenness_shortest_avg, -1., closeness_shortest_avg, -1., \
                   dist_agent_same_dir, dist_agent_opposite_dir, -1, -1

        extended_alternative_path = self._get_extended_shortest_path(agent, alternative_path)
        number_of_agents_on_path_same_dir_on_alternative_path, \
        number_of_agents_on_path_opposite_dir_on_alternative_path, dist_agent_same_dir_alternative, \
        dist_agent_opposite_dir_alternative = self._num_agents_on_path(handle, extended_alternative_path)

        betweenness_shortest_avg = np.mean([self.graph.nodes[n]['betweenness'] for n in shortest_path])
        closeness_shortest_avg = np.mean([self.graph.nodes[n]['closeness'] for n in shortest_path])

        betweenness_alternative_avg = np.mean([self.graph.nodes[n]['betweenness'] for n in alternative_path])
        closeness_alternative_avg = np.mean([self.graph.nodes[n]['closeness'] for n in alternative_path])

        return best_path, number_of_agents_on_path_same_dir, number_of_agents_on_path_opposite_dir, \
               alternative_path_length, number_of_agents_on_path_same_dir_on_alternative_path, \
               number_of_agents_on_path_opposite_dir_on_alternative_path, number_of_switches_on_shortest, betweenness_shortest_avg, \
               betweenness_alternative_avg, closeness_shortest_avg, closeness_alternative_avg, dist_agent_same_dir, \
               dist_agent_opposite_dir, dist_agent_same_dir_alternative, dist_agent_opposite_dir_alternative

    def segment_deadlock(self, handle: int) -> bool:
        """
        Checks if there is deadlock on segment

        :param handle: agent id
        :return: if agent is in deadlock or not
        """

        agent = self.agents[handle]

        if agent.Agent.status == RailAgentStatus.DONE or agent.Agent.status == RailAgentStatus.DONE_REMOVED:
            return False

        elif agent.Deadlock:
            return agent.Deadlock

        for id in self.agents:
            other_agent = self.agents[id]
            agent_pos = self._get_virtual_position(agent.Agent)
            other_agent_pos = self._get_virtual_position(other_agent.Agent)

            # if the agent has been stuck in the same position, check if there are other agents in his segment(s)
            if agent.Agent.speed_data[
                "position_fraction"] >= 1 and agent_pos == agent.Agent.old_position and handle != id:
                for next_n in agent.NextNodes:
                    agent_segment = self.graph[agent.CurrentNode][next_n]["segment"]
                    hashed_grid_cells = frozenset((cx, cy) for cx, cy, _ in agent_segment)
                    segment_attrs = self.segment_dict[hashed_grid_cells]["edges"]

                    # if (other_agent.CurrentNode, other_agent.NextNodes[0]) in segment_attrs and (
                    #         (agent.CurrentNode[0], agent.CurrentNode[1]) != (
                    #         other_agent.CurrentNode[0], other_agent.CurrentNode[1]) or
                    #         ((agent.CurrentNode[0], agent.CurrentNode[1]) == (
                    #                 other_agent.CurrentNode[0], other_agent.CurrentNode[1]) and
                    #          other_agent.Agent.speed_data[
                    #              "position_fraction"] >= 1.)) and other_agent.Agent.status == RailAgentStatus.ACTIVE:
                    #
                    #     self.segment_dict[hashed_grid_cells]["deadlock"] = True
                    #     rest_n_nodes = [n for n in agent.NextNodes if n != next_n]
                    #     for nn in rest_n_nodes:
                    #         agent_segment_n = self.graph[agent.CurrentNode][nn]["segment"]
                    #         hashed_grid_cells_n = frozenset((cx, cy) for cx, cy, _ in agent_segment_n)
                    #         self.segment_dict[hashed_grid_cells_n]["deadlock"] = True
                    #         self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                    #     return True

                    cell_transition = [i for i, v in
                                       enumerate(self.env.rail.get_transitions(*(agent_pos[0], agent_pos[1]), \
                                                                               self.agents[handle].Agent.direction)) if
                                       v == 1]

                    for transition in cell_transition:
                        dx, dy = self._get_coords(transition)

                        next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
                        dir = transition
                        visited = set()
                        while next_pos not in visited:
                            other_agent_next_pos_id = [a_id for a_id in self.agents if
                                                       next_pos == self._get_virtual_position(self.agents[a_id].Agent)]
                            if other_agent_next_pos_id:
                                if self.agents[other_agent_next_pos_id[0]].Deadlock or (self.agents[
                                                                                            other_agent_next_pos_id[
                                                                                                0]].Agent.direction != dir and
                                                                                        self.agents[
                                                                                            other_agent_next_pos_id[
                                                                                                0]].Agent.speed_data[
                                                                                            "position_fraction"] >= 1.):
                                    self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                                    for nn in agent.NextNodes:
                                        self.graph[agent.CurrentNode][nn]['weight'] = 5000
                                    self.shortest_paths.clear()
                                    return True
                                else:
                                    visited.add(next_pos)
                                    dir = self.agents[other_agent_next_pos_id[0]].Agent.direction
                                    dx, dy = self._get_coords(dir)
                                    next_pos = (next_pos[0] + dx, next_pos[1] + dy)
                            else:
                                break

            if handle != id and agent.Agent.status == RailAgentStatus.ACTIVE and other_agent.Agent.status == RailAgentStatus.ACTIVE and \
                    len(agent.NextNodes) == 1:
                agent_segment = self.graph[agent.CurrentNode][agent.NextNodes[0]]["segment"]
                segment_grid_cells = [(cx, cy) for cx, cy, _ in agent_segment]
                hashed_grid_cells = frozenset(segment_grid_cells)
                segment_attrs = self.segment_dict[hashed_grid_cells]["edges"]
                # if there is already deadlock on this segment return True
                if self.segment_dict[hashed_grid_cells]["deadlock"]:
                    self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                    self.graph[agent.CurrentNode][agent.NextNodes[0]]['weight'] = 5000
                    self.shortest_paths.clear()
                    return True
                # if any of the other agent paths is our segment
                other_agent_in_possible_segment = any(
                    (other_agent.CurrentNode, next_n) in segment_attrs for next_n in other_agent.NextNodes)
                if other_agent_in_possible_segment and (other_agent.CurrentNode[0], other_agent.CurrentNode[1]) \
                        != (agent.CurrentNode[0], agent.CurrentNode[1]):
                    agent_has_station_on_segment = any(e in segment_attrs for e in agent.EndEdges)
                    other_agent_has_station_on_segment = any(e in segment_attrs for e in other_agent.EndEdges)

                    agent_idx_on_segment = [i for i, ss in enumerate(segment_grid_cells) if agent_pos == ss][0]
                    other_agent_idx_on_segment = [i for i, ss in enumerate(segment_grid_cells) if
                                                  other_agent_pos == ss]

                    # if other agent was on switch and my id is before his, his edge, node attributes might not be updated
                    other_agent_on_segment = False if len(other_agent_idx_on_segment) == 0 else True

                    # if both agents have no station on segment
                    if not agent_has_station_on_segment and not other_agent_has_station_on_segment and other_agent_on_segment:
                        min_idx = min(agent_idx_on_segment, other_agent_idx_on_segment[0])
                        max_idx = max(agent_idx_on_segment, other_agent_idx_on_segment[0])
                        between_agents_segment = segment_grid_cells[min_idx + 1:max_idx]
                        # if there are not in-between cells between the agent, they are facing each other (deadlock) or away from each other
                        if len(between_agents_segment) == 0:
                            if segment_grid_cells[agent_idx_on_segment + 1] == other_agent_pos and \
                                    segment_grid_cells[other_agent_idx_on_segment[0] - 1] == agent_pos:
                                # if other agent is on switch, probably he is waiting
                                if other_agent.Agent.speed_data["position_fraction"] == 0 and \
                                        (other_agent.CurrentNode[0],
                                         other_agent.CurrentNode[1]) == other_agent_pos:
                                    continue
                                # return False
                                else:

                                    self.segment_dict[hashed_grid_cells]["deadlock"] = True
                                    self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                                    self.graph[agent.CurrentNode][agent.NextNodes[0]]['weight'] = 5000
                                    self.shortest_paths.clear()
                                    return True
                        # else:
                        #     return False
                        else:
                            if len(other_agent.NextNodes) == 1:
                                agent_cell_transition = \
                                    [i for i, v in
                                     enumerate(self.env.rail.get_transitions(*(agent_pos[0], agent_pos[1]), \
                                                                             agent.Agent.direction)) if
                                     v == 1][0]
                                other_agent_cell_transition = [i for i, v in enumerate(
                                    self.env.rail.get_transitions(*(other_agent_pos[0], other_agent_pos[1]), \
                                                                  other_agent.Agent.direction)) if v == 1][0]
                                dx, dy = self._get_coords(agent_cell_transition)
                                agent_next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
                                o_dx, o_dy = self._get_coords(other_agent_cell_transition)
                                other_agent_next_pos = (other_agent_pos[0] + o_dx, other_agent_pos[1] + o_dy)
                                if agent_next_pos in between_agents_segment and \
                                        other_agent_next_pos in between_agents_segment:
                                    self.segment_dict[hashed_grid_cells]["deadlock"] = True
                                    self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                                    self.graph[agent.CurrentNode][agent.NextNodes[0]]['weight'] = 5000
                                    self.shortest_paths.clear()
                                    return True
                        # else:
                        #     return False

                    # if current agent has station on segment
                    elif agent_has_station_on_segment and not other_agent_has_station_on_segment and other_agent_on_segment:
                        # print("Here1")
                        agent_idx_station = [i for i, ss in enumerate(segment_grid_cells) if agent.Agent.target == ss][
                            0]
                        other_agent_dist_to_station = other_agent_idx_on_segment[0] - agent_idx_station
                        if other_agent_dist_to_station <= 0:
                            # print("Here11")
                            self.segment_dict[hashed_grid_cells]["deadlock"] = True
                            self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                            self.graph[agent.CurrentNode][agent.NextNodes[0]]['weight'] = 5000
                            self.shortest_paths.clear()
                            return True

                    # if other agent has station on segment
                    elif not agent_has_station_on_segment and other_agent_has_station_on_segment and other_agent_on_segment:
                        # print("Here2")
                        other_agent_idx_station = \
                            [i for i, ss in enumerate(segment_grid_cells) if other_agent.Agent.target == ss][0]
                        agent_dist_to_other_station = other_agent_idx_station - agent_idx_on_segment
                        if agent_dist_to_other_station <= 0:
                            # print("Here21", agent_dist_to_other_station)
                            # if the other agent got stuck on switch
                            if other_agent.Agent.speed_data["position_fraction"] >= 1 or len(
                                    other_agent.NextNodes) == 1:
                                self.segment_dict[hashed_grid_cells]["deadlock"] = True
                                self.agents[handle] = self.agents[handle]._replace(Deadlock=True)
                                self.graph[agent.CurrentNode][agent.NextNodes[0]]['weight'] = 5000
                                self.shortest_paths.clear()
                                return True
        return False

    @staticmethod
    def _get_coords(direction: int) -> Tuple[int, int]:
        """
        Returns the dx and dy offsets based on the direction

        :param direction: Direction of agent
        :return: dx and dy offsets
        """
        # North direction
        if direction == 0:
            return -1, 0
        # East direction
        elif direction == 1:
            return 0, 1
        # South direction
        elif direction == 2:
            return 1, 0
        # West direction
        elif direction == 3:
            return 0, -1

    @staticmethod
    def draw_graph(G: nx.DiGraph):
        """
        Plots the given graph

        :param G: networkx Digraph
        :return: None
        """
        pos = nx.spring_layout(G, seed=10)
        nx.draw_networkx_nodes(G, pos, node_size=40)
        nx.draw_networkx_labels(G, pos, font_size=5)
        nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True, arrowsize=7)
        plt.show()

    def end_station_conflict(self, my_enter_time, my_exit_time, other_enter, other_exit, my_handle, other_handle):
        if my_exit_time - my_enter_time < other_exit - other_enter:
            # I am arriving at station
            l = self.graph.edges[self.agents[my_handle].EndEdges[0]]['weight'] - (my_exit_time - my_enter_time)
            other_exit -= l
            other_enter -= l
        elif my_exit_time - my_enter_time > other_exit - other_enter:
            l = self.graph.edges[self.agents[other_handle].EndEdges[0]]['weight'] - (other_exit - other_enter)
            my_enter_time -= l
            my_exit_time -= l
        # print("overlap", my_enter_time, my_exit_time, other_enter, other_exit,my_handle, other_handle)
        return self.get_overlap((my_enter_time, my_exit_time), (other_enter, other_exit))

    def find_edge_in_graph(self, segment):
        # print("finding edge", segment)
        for edge in self.graph.edges:
            if edge[0][0] == segment[0][0] and edge[0][1] == segment[0][1] and edge[1][0] == segment[1][0] and edge[1][
                1] == segment[1][1]:
                return edge
        return None

    def conflict_with_edge(self, segment_direction_agents, segment, my_enter_time, my_exit_time, handle):
        this_direc_seg = (segment[0], segment[1])
        opposite_direction = (this_direc_seg[1], this_direc_seg[0])
        if this_direc_seg not in segment_direction_agents and opposite_direction not in segment_direction_agents:
            return False

        for other_agent, enter_time, exit_time in segment_direction_agents.get(this_direc_seg, []):
            if enter_time == my_enter_time and enter_time != 0:
                return True

        opposite_direction = (this_direc_seg[1], this_direc_seg[0])

        for agent_other_dir, other_enter, other_exit in segment_direction_agents.get(opposite_direction, []):
            if self.get_overlap((my_enter_time, my_exit_time), (other_enter, other_exit)):
                if agent_other_dir != handle and self.if_conflict(handle, my_enter_time, my_exit_time, agent_other_dir,
                                                                 other_enter, other_exit, segment):
                    return True
        return False

    def does_agent_have_free_path(self, handle, segment_direction_agents):
        dist, visited, parent, start, best_path, end_node = self.shortest_paths[handle]
        shortest_path = self._construct_shortest_path(parent, end_node)
        cur_node = self.agents[handle].CurrentNode
        starting_edge_no_dir = ((cur_node[0], cur_node[1]), (shortest_path[0][0], shortest_path[0][1]))

        if self.env.agents[handle].status==RailAgentStatus.READY_TO_DEPART:
            addition = 1
        else:
            addition=0

        if self.conflict_with_edge(segment_direction_agents, starting_edge_no_dir, 0, dist[shortest_path[0]] + addition, handle):
            return False

        for idx, segment in enumerate(shortest_path):
            if idx == len(shortest_path) - 1:
                end_dist = best_path
                end_seg = self.find_end_segment(handle, shortest_path[idx])
            else:
                end_dist = dist[shortest_path[idx + 1]]
                end_seg = shortest_path[idx + 1]

            switch_no_dir = (segment[0], segment[1])
            segment_not_dir = (switch_no_dir, (end_seg[0], end_seg[1]))

            if self.conflict_with_edge(segment_direction_agents, segment_not_dir, dist[segment] + addition, end_dist + addition, handle):
                return False
        return True

    def find_end_segment(self, handle, last_edge):
        end_edges_for_agent = self.agents[handle].EndEdges
        for edge in end_edges_for_agent:
            if edge[0] == last_edge:
                return edge[1]
        return None

    def build_segment_timestamp_dict(self, segment_direction_agents, list_agents):

        for agent_handle in list_agents:
            agent = self.env.agents[agent_handle]
            if agent.handle not in list_agents:
                continue
            dist, visited, parent, start, best_path, end_node = self.shortest_paths[agent_handle]
            if end_node == 0 or end_node == None:
                continue

            if agent.status==RailAgentStatus.READY_TO_DEPART:
                addition = 1
            else:
                addition=0

            shortest_path = self._construct_shortest_path(parent, end_node)
            cur_node = self.agents[agent_handle].CurrentNode
            next_node = shortest_path[0]
            starting_edge_no_dir = ((cur_node[0], cur_node[1]), (next_node[0], next_node[1]))
            if cur_node != next_node:
                segment_direction_agents.setdefault(starting_edge_no_dir, set([]))
                segment_direction_agents[starting_edge_no_dir].add((agent_handle, 0, dist[shortest_path[0]] + addition))

            for idx, switch in enumerate(shortest_path):
                switch_no_dir = (switch[0], switch[1])

                if idx == len(shortest_path) - 1:
                    end_dist = best_path
                    end_seg = self.find_end_segment(agent_handle, shortest_path[idx])
                else:
                    end_dist = dist[shortest_path[idx + 1]]
                    end_seg = shortest_path[idx + 1]

                segment_not_dir = (switch_no_dir, (end_seg[0], end_seg[1]))
                segment_direction_agents.setdefault(segment_not_dir, set([]))
                segment_direction_agents[segment_not_dir].add((agent_handle, dist[switch] + addition, end_dist + addition))
        return segment_direction_agents

    @staticmethod
    def find_equal_starting_points(list_agents):
        equal_start = []
        for i in range(len(list_agents)):
            for j in range(len(list_agents)):
                agent1, agent2 = list_agents[i][0], list_agents[j][0]
                dist1, dist2 = list_agents[i][1], list_agents[j][1]
                if agent1 != agent2 and dist1 == dist2 and dist1 != 0 and dist2 != 0:
                    equal_start.append((agent1, agent2))
        return equal_start

    @staticmethod
    def get_overlap(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) >= 0

    def extract_path(self, path, handle):
        if self.env.agents[handle].position is None:
            virtual_position = self.env.agents[handle].initial_position
            starting_path=[None]
        else:
            virtual_position = self.env.agents[handle].position
            starting_path = []

        if virtual_position in path:
            index_position = path.index(virtual_position)
        else:
            index_position = 0
            starting_path = []
        starting_path.extend(path[index_position:])
        return starting_path

    def if_conflict(self, agent1, dist_interval1_start, dist_interval1_end, agent2, dist_interval2_start,
                    dist_interval2_end, segment):

        if self.get_overlap((dist_interval1_start, dist_interval1_end), (dist_interval2_start, dist_interval2_end)) > 0:
            if dist_interval1_start != 0 and dist_interval2_start != 0:
                if dist_interval1_end - dist_interval1_start != dist_interval2_end - dist_interval2_start:
                    if self.end_station_conflict(dist_interval1_start, dist_interval1_end, dist_interval2_start,
                                                 dist_interval2_end, agent1, agent2):
                        # print("end station conflict", agent1, agent2, segment)
                        return True
                else:
                    # print("regular conflicts", agent1, agent2, segment)
                    return True
            else:
                time_position = {}
                time_agent = {}
                if dist_interval1_start == 0:
                    agent_segment = self.graph[self.agents[agent1].CurrentNode][self.agents[agent1].NextNodes[0]]["segment"]
                    segment_grid_cells = [(cx, cy) for cx, cy, _ in agent_segment]
                    path1 = self.extract_path(segment_grid_cells[:], agent1)
                    path2 = self.extract_path(segment_grid_cells[::-1], agent2)

                else:
                    agent_segment = self.graph[self.agents[agent2].CurrentNode][self.agents[agent2].NextNodes[0]]["segment"]
                    segment_grid_cells = [(cx, cy) for cx, cy, _ in agent_segment]
                    path1 = self.extract_path(segment_grid_cells[:], agent2)
                    path2 = self.extract_path(segment_grid_cells[::-1], agent1)


                range_len = min(len(path1), len(path2))

                for i in range(range_len):
                    if path1[i] == path2[i]:
                        return True
                    if i + 1 < range_len:
                        if path1[i] == path2[i+1] and path2[i] == path1[i+1]:
                            return True

        return False

    def find_intersecting_intervals(self, list_intervals_dir1, list_intervals_dir2, segment):
        conflicts = []
        for (agent1, dist_interval1_start, dist_interval1_end) in list_intervals_dir1:
            for (agent2, dist_interval2_start, dist_interval2_end) in list_intervals_dir2:
                if self.get_overlap((dist_interval1_start, dist_interval1_end),
                                    (dist_interval2_start, dist_interval2_end)) > 0:
                    if agent1 != agent2 and self.if_conflict(agent1, dist_interval1_start, dist_interval1_end, agent2,
                                                             dist_interval2_start, dist_interval2_end, segment):
                        conflicts.append((agent1, agent2))
        return conflicts

    def build_wait_for_graph(self, segment_direction_agents, set_to_depart):
        wait_for_graph = {a.handle: set([]) for a in self.env.agents if
                          a.status == RailAgentStatus.ACTIVE or a.handle in set_to_depart}
        for segment in segment_direction_agents:
            for wait_for_tuple in self.find_equal_starting_points(list(segment_direction_agents[segment])):
                wait_for_graph[wait_for_tuple[0]].add(wait_for_tuple[1])
                # print("Added ", wait_for_tuple, "in same dir", segment)

            opposite_direc = (segment[1], segment[0])
            if opposite_direc in segment_direction_agents:
                for conflict in self.find_intersecting_intervals(list(segment_direction_agents[segment]),
                                                                 list(segment_direction_agents[opposite_direc]),
                                                                 opposite_direc):
                    wait_for_graph[conflict[0]].add(conflict[1])
                    # ("Added ", conflict, "in opposite dir", segment)
        return wait_for_graph

    @staticmethod
    def connected_components(graph):
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

    def color_component(self, component, graph, lies_on_path_dict):
        blocked_agents = set([])
        blocked_path_agents = set([])
        lies_on_path_component = {}
        for a in component:
            if a not in lies_on_path_dict:
                continue
            in_same_component = [other for other in lies_on_path_dict[a] if other in component]

            if len(in_same_component) > 0:
                lies_on_path_component[a] = set(in_same_component)
                blocked_path_agents = blocked_path_agents.union(in_same_component)

        set_component = set(component)
        non_blocked_agents = set_component.difference(lies_on_path_component.keys())
        non_blocked_agents = non_blocked_agents.intersection(blocked_path_agents)
        colors = {}

        # go with the ones that do not have blocked path

        for agent in non_blocked_agents:
            has_blocked_neighb = all([0 for neigh in graph[agent] if neigh in blocked_agents])
            if agent not in colors and agent in component and has_blocked_neighb:
                colors[agent] = 1
                for vertex in graph[agent]:
                    colors[vertex] = 0

        # first, go with non integer fractions

        # then, go prev_going

        elem_in_component = len(component)
        prev_going = set([a for a in self.prev_priorities if self.prev_priorities[a] == 1])
        for agent in prev_going:
            if agent not in set_component:
                continue
            a = self.env.agents[agent]
            agent_group_id = (a.initial_position, a.target)

            get_same_group = self.get_same_group_start_and_end()
            for same_group_agent in get_same_group[agent_group_id]:
                if same_group_agent in set_component:
                    has_blocked_neighb = all([0 for neigh in graph[same_group_agent] if neigh in blocked_agents])
                    if same_group_agent not in colors and agent and has_blocked_neighb:
                        colors[agent] = 1
                        for vertex in graph[agent]:
                            colors[vertex] = 0

        # then, go prev_going
        # print(prev_going, "prev_going")
        elem_in_component = len(component)
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

            if largest is None:
                break
            colors[largest] = 1

            for vertex in graph[largest]:
                colors[vertex] = 0
        for elem in component:
            colors.setdefault(elem, 0)
        return colors

    def get_same_group_ready_to_depart(self) -> dict:
        group_depart = {}
        for agent in self.env.agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                group_depart.setdefault(agent.initial_position, set([]))
                group_depart[agent.initial_position].add(agent.handle)
        return group_depart

    def get_same_group_start_and_end(self) -> dict:
        group_depart = {}
        for agent in self.env.agents:
            t = (agent.initial_position, agent.target)
            group_depart.setdefault(t, set([]))
            group_depart[t].add(agent.handle)
        return group_depart

    def choose_agent_to_depart_for_group(self, group_depart, segment_dir):
        sorted_agents = [(self.shortest_paths[agent][4], agent) for agent in group_depart if
                         self.does_agent_have_free_path(agent, segment_dir)]
        if len(sorted_agents) == 0:
            return None
        return sorted_agents[0][1]

    def add_ready_to_depart_agent_to_priority(self, segment_direction_agents):
        set_to_depart = set([])
        set_same_start_ready_to_depart = self.get_same_group_ready_to_depart()
        for group_depart in set_same_start_ready_to_depart:
            agent_to_depart = self.choose_agent_to_depart_for_group(set_same_start_ready_to_depart[group_depart],
                                                                    segment_direction_agents)
            if agent_to_depart is not None:
                set_to_depart.add(agent_to_depart)
                tmp_set = set([agent_to_depart])
                segment_direction_agents = self.build_segment_timestamp_dict(segment_direction_agents, tmp_set)

            for other_agent_id in set_same_start_ready_to_depart[group_depart]:
                if other_agent_id != agent_to_depart:
                    self.priorities[other_agent_id] = 0
        return segment_direction_agents, set_to_depart

    def lies_on_path(self, segment_direction_agents, set_to_depart):
        blocked_path = {}
        for agent in self.env.agents:
            if agent.status != RailAgentStatus.ACTIVE and agent.handle not in set_to_depart:
                continue
            dist, visited, parent, start, best_path, end_node = self.shortest_paths[agent.handle]
            if end_node == 0 or end_node == None:
                continue

            shortest_path = self._construct_shortest_path(parent, end_node)
            agent_dict = self.agents[agent.handle]
            all_current_nodes = [
                ((agent_dict.CurrentNode[0], agent_dict.CurrentNode[1]), (other_switch[0], other_switch[1])) for
                other_switch in agent_dict.NextNodes]

            for node in all_current_nodes:
                if node in segment_direction_agents:

                    for blocked_agent, other_enter, other_exit in list(segment_direction_agents[node]):
                        if blocked_agent != agent.handle:
                            if other_enter > 0 or dist[shortest_path[0]] < other_exit:
                                blocked_path.setdefault(blocked_agent, set([]))
                                blocked_path[blocked_agent].add(agent.handle)
                                # blocked_path[agent.handle].add(blocked_agent)

                opposite_segment = (node[1], node[0])
                if opposite_segment in segment_direction_agents:
                    for blocked_agent, _, _ in list(segment_direction_agents[opposite_segment]):
                        blocked_path.setdefault(blocked_agent, set([]))
                        blocked_path[blocked_agent].add(agent.handle)

        return blocked_path

    def calculate_priorities(self):
        self.prev_priorities = copy.deepcopy(self.priorities)
        self.priorities = {}
        segment_direction_agents = {}
        active_agents = [a.handle for a in self.env.agents if a.status == RailAgentStatus.ACTIVE]
        segment_direction_agents = self.build_segment_timestamp_dict(segment_direction_agents, active_agents)
        segment_direction_agents, set_to_depart = self.add_ready_to_depart_agent_to_priority(segment_direction_agents)
        # print("set to depart", set_to_depart)
        # self.pretty_print_segment(segment_direction_agents)
        graph = self.build_wait_for_graph(segment_direction_agents, set_to_depart)
        # print("wait for", graph)
        lies_on_path_dict = self.lies_on_path(segment_direction_agents, set_to_depart)
        components = self.connected_components(graph)
        # print("components,", components)
        for c in components:
            colors = self.color_component(c, graph, lies_on_path_dict)
            # print("colors", colors)
            for a in colors:
                self.priorities[a] = colors[a]
        # print("priorities", self.priorities)

    def pretty_print_segment(self, segment_dir_dict):
        for agent in self.env.agents:
            agent_pairs = []
            for segment in segment_dir_dict:
                for pair in segment_dir_dict[segment]:
                    if pair[0] == agent.handle:
                        agent_pairs.append([segment, pair[0], pair[1], pair[2]])

            import operator
            s_l = sorted(agent_pairs, key=operator.itemgetter(2))
            if len(s_l) > 0:
                print(agent.handle, s_l)

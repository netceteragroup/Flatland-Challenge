import collections

import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.ordered_set import OrderedSet
from typing import Optional, List, Dict


class MyTreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """
    Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                          'dist_other_target_encountered '
                                          'dist_other_agent_encountered '
                                          'dist_potential_conflict '
                                          'dist_unusable_switch '
                                          'dist_to_next_branch '
                                          'dist_min_to_target '
                                          'num_agents_same_direction '
                                          'num_agents_opposite_direction '
                                          'num_agents_malfunctioning '
                                          'speed_min_fractional '
                                          'num_agents_ready_to_depart '
                                          'total_cells '
                                          'total_switch '
                                          'total_switches_neighbors '
                                          'index_comparision '
                                          'first_switch_free '
                                          'first_switch_neighbor '
                                          'childs')

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 17
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None

        self.switches_list = []
        self.switches_neightbors_list = []

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}

    def set_switch_and_pre_switch(self, switch_list, pre_switch_list):
        self.switches_list = switch_list
        self.switches_neightbors_list = pre_switch_list

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
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

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

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
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        root_node_observation = MyTreeObsForRailEnv.Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                                         dist_other_agent_encountered=0, dist_potential_conflict=0,
                                                         dist_unusable_switch=0, dist_to_next_branch=0,
                                                         dist_min_to_target=distance_map[
                                                             (handle, *agent_virtual_position,
                                                              agent.direction)],
                                                         num_agents_same_direction=0, num_agents_opposite_direction=0,
                                                         num_agents_malfunctioning=agent.malfunction_data[
                                                             'malfunction'],
                                                         speed_min_fractional=agent.speed_data['speed'],
                                                         num_agents_ready_to_depart=0,
                                                         total_cells=0,
                                                         total_switch=0,
                                                         total_switches_neighbors=0,
                                                         index_comparision=0,
                                                         first_switch_free=0,
                                                         first_switch_neighbor=0,
                                                         childs={})

        visited = OrderedSet()
        self.opposite = []

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited
        self.opposite = []

        return root_node_observation

    def _explore_branch(self, handle, position, direction, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = 0
        other_agent_encountered = 0
        other_target_encountered = 0
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        total_cells = 0
        total_switch = 0
        total_switches_neighbors = 0
        other_agent_ready_to_depart_encountered = 0
        index_comparision = 0
        first_switch_free = 0
        first_switch_neighbor = 0
        while exploring:
            total_cells += 1
            if len(self.switches_list) > 0:
                if position in self.switches_list:
                    total_switch += 1
            if len(self.switches_neightbors_list) > 0:
                if position in self.switches_neightbors_list:
                    total_switches_neighbors += 1

            opp_a = self.env.agent_positions[position]
            if opp_a > -1 and opp_a != handle and index_comparision == 0:
                index_comparision = 2.0 * int(opp_a < handle) - 1.0

            if first_switch_free == 0:
                if position in self.switches_list:
                    if opp_a > -1:
                        first_switch_free = 1.0
                    else:
                        first_switch_free = -1.0

            if first_switch_neighbor == 0:
                if position in self.switches_neightbors_list:
                    if opp_a > -1:
                        first_switch_neighbor = 1.0
                    else:
                        first_switch_neighbor = -1.0

            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    # other_agent_same_direction += self.location_has_agent_direction.get((position, direction), 0)

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

            # possible_transitions = self.env.rail.get_transitions(*position, direction)
            x = self.env.agent_positions[position]
            if x != -1:
                if x != handle:
                    other_agent_encountered += 1.0
            if position in self.location_has_target and position != agent.target:
                other_target_encountered += 1.0

            if position == agent.target:
                own_target_encountered = 1.0

            num_transitions = 1  # np.count_nonzero(possible_transitions)
            if num_transitions == 1:
                new_direction_me = direction  # np.argmax(possible_transitions)
                new_cell_me = position  # get_new_position(position, new_direction_me)
                a = self.env.agent_positions[new_cell_me]
                if a != -1 and a != handle:
                    opp_agent = self.env.agents[a]
                    # look one step forward
                    possible_transitions = self.env.rail.get_transitions(*opp_agent.position, opp_agent.direction)
                    if possible_transitions[new_direction_me] == 0:
                        self.opposite.append(position)
                        other_agent_opposite_direction += 1
                        break  # path is broken
                    else:
                        other_agent_same_direction += 1

            # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        node = MyTreeObsForRailEnv.Node(dist_own_target_encountered=own_target_encountered,
                                        dist_other_target_encountered=other_target_encountered,
                                        dist_other_agent_encountered=other_agent_encountered,
                                        dist_potential_conflict=potential_conflict,
                                        dist_unusable_switch=unusable_switch,
                                        dist_to_next_branch=dist_to_next_branch,
                                        dist_min_to_target=dist_min_to_target,
                                        num_agents_same_direction=other_agent_same_direction,
                                        num_agents_opposite_direction=other_agent_opposite_direction,
                                        num_agents_malfunctioning=malfunctioning_agent,
                                        speed_min_fractional=min_fractional_speed,
                                        num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                                        total_cells=total_cells,
                                        total_switch=total_switch,
                                        total_switches_neighbors=total_switches_neighbors,
                                        index_comparision=index_comparision,
                                        first_switch_free=first_switch_free,
                                        first_switch_neighbor=first_switch_neighbor,
                                        childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)

        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited

    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)

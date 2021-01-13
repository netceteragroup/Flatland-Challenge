from typing import Optional, List, Dict, Union, Tuple

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.ordered_set import OrderedSet

from envs.flatland.observations import Observation, register_obs  # noqa
from itertools import combinations
import collections

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus

from flatland.core.grid.grid4_utils import get_new_position


@register_obs("localConflict")
class LocalConflictObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = LocalConflictObsForRailEnvRLLibWrapper(
            LocalConflictObsForRailEnv(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(
                    config['shortest_path_max_depth']),
                n_local=config['n_local'])
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        num_features = self._builder.observation_dim
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,))


class LocalConflictObsForRailEnvRLLibWrapper(ObservationBuilder):

    """
    The information is for each agent but uses the full set of
    observations for all agents to come up with set of local
    (Default: 5) most conflicting agents.

    The observation set is based on the current agent and these local
    identified agents. We also information about conflicts.
    """

    def __init__(self, local_conflict_obs_builder: TreeObsForRailEnv):
        super().__init__()
        self._builder = local_conflict_obs_builder
        # To cache calculated agent states
        # This is only computed once and reused for all other agents
        self.agent_states: Optional[Dict] = None

    @property
    def observation_dim(self):
        return self._builder.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        n_agents = self._builder.get_number_of_agents()
        obs = []
        for handle in range(n_agents):
            obs.append(self._builder.get(handle))

        if not self.agent_states:
            self.agent_states = create_agent_states(
                obs, self._builder.predictor.max_depth)

        return self.agent_states[handle] \
            if obs is not None else obs

    def get_many(self, handles: Optional[List[int]] = None):

        all_agent_observations = self._builder.get_many(handles)
        obs = dict()
        if handles is None:
            handles = []
        for k in handles:
            if not self.agent_states:
                self.agent_states = create_agent_states(
                    all_agent_observations, self._builder.predictor.max_depth)
            obs[k] = self.agent_states[k]

        return obs

    def set_env(self, env):
        self._builder.set_env(env)


class LocalConflictObsForRailEnv(TreeObsForRailEnv):
    """
    LocalConflict object made from TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv.


    For details about the features in the observation
    see the get() function.
    We normalise all observations based on the grid size
    """
    Node = collections.namedtuple('Node', 'distance_target '
                                          'observation_shortest '
                                          'observation_next_shortest '
                                          'extra_distance '
                                          'malfunction '
                                          'malfunction_rate '
                                          'next_malfunction '
                                          'nr_malfunctions '
                                          'speed '
                                          'position_fraction '
                                          'transition_action_on_cellexit '
                                          'num_transitions '
                                          'moving '
                                          'status '
                                          'action_required '
                                          'width '
                                          'height '
                                          'n_agents '
                                          'predictions '
                                          'predicted_pos')

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None,
                 n_local: int = 5):
        super().__init__(max_depth, predictor)
        self.observation_dim = 1 + 3 * (n_local - 1) + 22 * n_local

    def reset(self):
        super().reset()

    def get_many(self, handles: Optional[List[int]] = None):

        observations = super().get_many(handles)
        return observations

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(
                *agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(
                *agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions
        # are available;
        # organize them as [left, forward, right], relative to
        # the current orientation
        # If only one transition is possible, the forward branch is
        # aligned with it.
        distance_map = self.env.distance_map.get()
        max_distance = self.env.width + self.env.height
        # max_steps = int(4 * 2 * (20 + self.env.height + self.env.width))

        visited = OrderedSet()
        for _idx in range(10):
            # Check if any of the other prediction overlap
            # with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))

        # This variable will be access by the renderer to
        # visualize the observation
        self.env.dev_obs_dict[handle] = visited

        # min_distance stores the distance to target in shortest path
        # and any alternate path if exists
        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(
                    agent_virtual_position, direction)
                min_distances.append(
                    distance_map[handle, new_position[0],
                                 new_position[1], direction])
            else:
                min_distances.append(np.inf)

        if num_transitions == 1:
            observation1 = [0, 1, 0]
            observation2 = observation1

        elif num_transitions == 2:
            idx = np.argpartition(np.array(min_distances), 2)
            observation1 = [0, 0, 0]
            observation1[idx[0]] = 1

            observation2 = [0, 0, 0]
            observation2[idx[1]] = 1

        min_distances = np.sort(min_distances)
        incremental_distances = np.diff(np.sort(min_distances))
        incremental_distances[incremental_distances == np.inf] = 0
        incremental_distances[np.isnan(incremental_distances)] = 0

        distance_target = distance_map[(handle, *agent_virtual_position,
                                        agent.direction)]

        root_node_observation = LocalConflictObsForRailEnv.Node(
            distance_target=distance_target / max_distance,
            observation_shortest=observation1,
            observation_next_shortest=observation2,
            extra_distance=incremental_distances[
                0] / max_distance,
            malfunction=agent.malfunction_data[
                'malfunction'] / max_distance,
            malfunction_rate=agent.malfunction_data[
                'malfunction_rate'],
            next_malfunction=agent.malfunction_data[
                'next_malfunction'] / max_distance,
            nr_malfunctions=agent.malfunction_data[
                'nr_malfunctions'],
            speed=agent.speed_data['speed'],
            position_fraction=agent.speed_data[
                'position_fraction'],
            transition_action_on_cellexit=agent.speed_data[
                'transition_action_on_cellexit'],
            num_transitions=num_transitions,
            moving=agent.moving,
            status=agent.status,
            action_required=action_required(agent),
            width=self.env.width,
            height=self.env.height,
            n_agents=self.get_number_of_agents(),
            predictions=self.predictions[handle],
            predicted_pos=self.predicted_pos)

        return root_node_observation

    def get_number_of_agents(self):
        return self.env.get_num_agents()


def create_agent_states(obs: Union[Dict, List],
                        max_depth: int, n_local: int = 5) -> Dict:
    """
    Identifies local agent conflicts and adds information from
    conflict prediction matrix. For more details refer to the
    observation section in the README.md file.
    """
    n_agents = len(obs)
    x_dim = 0
    y_dim = 0
    for i in range(n_agents):
        if obs[i] is not None:
            custom_observations = obs[i]
            x_dim = custom_observations.width
            y_dim = custom_observations.height
            break

    local_agent_states_all = dict()

    distance_target = np.ones(n_agents)
    extra_distance = np.zeros(n_agents)
    malfunction = np.zeros(n_agents)
    malfunction_rate = np.zeros(n_agents)
    next_malfunction = np.zeros(n_agents)
    nr_malfunctions = np.zeros(n_agents)
    speed = np.zeros(n_agents)
    position_fraction = np.zeros(n_agents)
    transition_action_on_cellexit = np.zeros(n_agents)
    num_transitions = np.zeros(n_agents)
    moving = np.zeros(n_agents)
    status = np.zeros(n_agents)
    info_action_required = np.zeros(n_agents)

    for i in range(n_agents):
        if obs[i] is not None:
            custom_observations = obs[i]
            distance_target[i] = custom_observations.distance_target
            extra_distance[i] = custom_observations.extra_distance
            malfunction[i] = custom_observations.malfunction
            malfunction_rate[i] = custom_observations.malfunction_rate
            next_malfunction[i] = custom_observations.next_malfunction
            nr_malfunctions[i] = custom_observations.nr_malfunctions
            speed[i] = custom_observations.speed
            position_fraction[i] = custom_observations.position_fraction
            transition_action_on_cellexit[i] = \
                custom_observations.transition_action_on_cellexit
            num_transitions[i] = int(custom_observations.num_transitions > 1)
            moving[i] = int(custom_observations.moving)
            status[i] = int(custom_observations.status > 0)
            info_action_required[i] = int(custom_observations.action_required)

    predicted_pos = custom_observations.predicted_pos
    agent_conflicts_count_path, agent_conflicts_step_path,\
        agent_total_step_conflicts = get_agent_conflict_prediction_matrix(
            n_agents, max_depth, predicted_pos)

    # Normalise based on average grid dimensions
    avg_dim = (x_dim * y_dim) ** 0.5
    depth = int(n_local * avg_dim / n_agents)

    agent_conflict_steps = min(max_depth - 1, depth)

    agent_conflicts = agent_conflicts_step_path[agent_conflict_steps]
    agent_conflicts_avg_step_count = np.average(
        agent_total_step_conflicts) / n_agents

    for i in range(n_agents):
        if obs[i] is not None:
            n_upd_local = min(n_local, n_agents - 1)
            if n_upd_local < n_local:
                n_pad = n_local - n_upd_local
                ls_other_local_agents = np.argpartition(
                    agent_conflicts[i, :], n_upd_local)[:n_upd_local - 1]
                for j in range(n_pad):
                    ls_other_local_agents = np.hstack(
                        [ls_other_local_agents, i])
            else:
                ls_other_local_agents = np.argpartition(
                    agent_conflicts[i, :], n_local)[:n_local - 1]
            ls_local_agents = np.hstack([i, ls_other_local_agents])
            local_agent_states = np.hstack(
                [distance_target[ls_local_agents],
                 extra_distance[ls_local_agents]])

            local_agent_states = np.hstack(
                [local_agent_states, info_action_required[ls_local_agents]])
            local_agent_states = np.hstack([local_agent_states,
                                            agent_conflicts_step_path[0]
                                            [i, ls_other_local_agents],
                                            agent_conflicts_step_path[1]
                                            [i, ls_other_local_agents],
                                            agent_conflicts_step_path[2]
                                            [i, ls_other_local_agents]])
            local_agent_states = np.hstack([local_agent_states,
                                            agent_conflicts_count_path[0]
                                            [ls_local_agents],
                                            agent_conflicts_count_path[1]
                                            [ls_local_agents],
                                            agent_conflicts_count_path[2]
                                            [ls_local_agents]])

            local_agent_states = np.hstack(
                [local_agent_states, malfunction[ls_local_agents],
                 malfunction_rate[ls_local_agents],
                 next_malfunction[ls_local_agents],
                 nr_malfunctions[ls_local_agents], speed[ls_local_agents],
                 position_fraction[ls_local_agents],
                 transition_action_on_cellexit[ls_local_agents],
                 num_transitions[ls_local_agents],
                 moving[ls_local_agents], status[ls_local_agents]])

            for j in ls_local_agents:
                if obs[j] is None:
                    local_agent_states = np.hstack(
                        [local_agent_states, [0, 0, 0]])
                    local_agent_states = np.hstack(
                        [local_agent_states, [0, 0, 0]])
                else:
                    local_agent_states = np.hstack(
                        [local_agent_states, obs[j].observation_shortest])
                    local_agent_states = np.hstack(
                        [local_agent_states, obs[j].observation_next_shortest])

            local_agent_states = np.hstack(
                [local_agent_states, agent_conflicts_avg_step_count])
            local_agent_states_all[i] = local_agent_states
    return local_agent_states_all


def get_agent_conflict_prediction_matrix(n_agents, max_depth, predicted_pos
                                         ) -> Tuple[List, List, List]:
    '''
    Calculates the agent conflict step path and agent conflict count path
    and the agent total conflict steps
    For more details refer to the observation section in the README.md file.
    '''
    agent_total_step_conflicts = []
    agent_conflicts_step_path = []
    agent_conflicts_count_path = []
    values = []
    counts = []
    agent_conflicts_step = max_depth * np.ones((n_agents, n_agents))

    for i in range(max_depth):
        step = i + 1
        pos = predicted_pos[i]
        val, count = np.unique(pos, return_counts=True)
        if val[0] == -1:
            val = val[1:]
            count = count[1:]
        values.append(val)
        counts.append(count)

        counter = np.zeros(n_agents)
        agent_conflicts_count = np.zeros(n_agents)

        for j, curVal in enumerate(val):
            curCount = count[j]
            if curCount > 1:
                idxs = np.argwhere(pos == curVal)
                lsIdx = [int(x) for x in idxs]
                combs = list(combinations(lsIdx, 2))
                for k, comb in enumerate(combs):
                    counter[comb[0]] += 1
                    counter[comb[1]] += 1
                    agent_conflicts_count[comb[0]] = counter[comb[0]]
                    agent_conflicts_count[comb[1]] = counter[comb[1]]
                    agent_conflicts_step[comb[0], comb[1]] = min(
                        step, agent_conflicts_step[comb[0], comb[1]])
                    agent_conflicts_step[comb[1], comb[0]] = min(
                        step, agent_conflicts_step[comb[1], comb[0]])

        agent_conflicts_step_current = agent_conflicts_step / max_depth
        agent_conflicts_step_path.append(agent_conflicts_step_current)
        agent_conflicts_count = agent_conflicts_count / n_agents
        agent_conflicts_count_path.append(agent_conflicts_count)

    for i in range(n_agents):
        agent_total_step_conflicts.append(
            sum(agent_conflicts_step_current[i, :]))

    return agent_conflicts_count_path, agent_conflicts_step_path,\
        agent_total_step_conflicts


def action_required(agent):
    """
    Check if an agent needs to provide an action

    Parameters
    ----------
    agent: RailEnvAgent
    Agent we want to check

    Returns
    -------
    True: Agent needs to provide an action
    False: Agent cannot provide an action
    """
    return (agent.status == RailAgentStatus.READY_TO_DEPART or (
        agent.status == RailAgentStatus.ACTIVE and
        np.isclose(agent.speed_data['position_fraction'], 0.0,
                   rtol=1e-03)))


def strategy_action_map(action, observation_shortest,
                        observation_next_shortest):
    """
    convert action space from 0-2 to 0-4
    observation_shortest and observation_next_shortest represent a
    3-size vector for the actions L,F and R.
    If no alternate path exists both of the values would be same
    E.g. observation_shortest = [0, 1, 0] refers to Forward (F) shortest path
    observation_next_shortest = [0, 0, 1] refers to taking action
    Right (R) for an alternate route
    """
    if action == 2:
        return 4
    elif action == 0:

        return np.argmax(observation_shortest) + 1
    elif action == 1:
        return np.argmax(observation_next_shortest) + 1


def action_strategy_map(action, observation_shortest,
                        observation_next_shortest, moving):
    """
    convert action space from 0-4 to 0-2 representing shortest path, deviate
    and stop
    Refer to the strategy_action_map method for observation arguments
    """
    if action == np.argmax(observation_shortest) + 1:
        return 0
    elif action == np.argmax(observation_next_shortest) + 1:
        return 1
    elif action == 0:
        if moving:
            if np.argmax(observation_shortest) == 1:
                return 0
            elif np.argmax(observation_shortest) == 1:
                return 1
        else:
            return 2
    elif action == 4:
        return 2
    else:
        return 0

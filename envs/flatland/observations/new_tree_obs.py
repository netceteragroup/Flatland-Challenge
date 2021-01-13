import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from typing import Optional, List

from envs.flatland.observations import Observation, register_obs
from envs.flatland.observations.new_tree_obs_builder import MyTreeObsForRailEnv as TreeObsForRailEnv


@register_obs("new_tree")
class TreeObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = TreeObsForRailEnvRLLibWrapper(
            TreeObsForRailEnv(
                max_depth=config['max_depth'],
                predictor=ShortestPathPredictorForRailEnv(config['shortest_path_max_depth'])
            )
        )

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        # TODO compute properly
        """
        num_features_per_node = self._builder.observation_dim
        nr_nodes = 0
        for i in range(self.config['max_depth'] + 1):
            nr_nodes += np.power(4, i)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features_per_node * nr_nodes,))
        """
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(151,))


def _split_node_into_feature_groups(node: TreeObsForRailEnv.Node, dist_min_to_target: int) -> (np.ndarray, np.ndarray,
                                                                                               np.ndarray):
    data = np.zeros(3)
    distance = np.zeros(1)
    agent_data = np.zeros(3)

    # np.seterr('raise')
    data[0] = 2.0 * (node.dist_other_agent_encountered > 0) - 1.0
    data[1] = 2.0 * (node.dist_other_target_encountered > 0) - 1.0
    data[2] = 2.0 * (node.dist_own_target_encountered > 0) - 1.0
    # data[3] = 2.0 * ((dist_min_to_target - node.dist_potential_conflict) > 0) - 1.0
    # data[4] = 2.0 * ((dist_min_to_target - node.dist_unusable_switch) > 0) - 1.0
    # data[5] = 2.0 * ((dist_min_to_target - node.dist_to_next_branch) > 0) - 1.0

    if dist_min_to_target != np.inf:
        distance[0] = 2.0 * ((dist_min_to_target - node.dist_min_to_target) > 0) - 1.0

    agent_data[0] = 2.0 * int(node.num_agents_opposite_direction > 0) - 1.0
    agent_data[1] = 2.0 * int(node.num_agents_same_direction > 0) - 1.0
    agent_data[2] = node.index_comparision
    # agent_data[3] = 2.0 * int(node.total_cells > 0) - 1.0
    # agent_data[4] = node.first_switch_free
    # agent_data[5] = node.first_switch_neighbor
    # agent_data[6] = 2.0 * int(node.total_cells == 1) - 1.0
    # agent_data[5] = 2.0 * int(node.total_cells == 2) - 1.0
    # agent_data[6] = 2.0 * int(node.total_cells > 2) - 1.0
    # agent_data[4] = 2.0 * int(node.total_cells <= 1) - 1.0
    # agent_data[5] = 2.0 * int(node.total_cells - node.total_switch > 0) - 1.0
    # agent_data[6] = 2.0 * int(node.total_cells - node.total_switches_neighbors > 0) - 1.0
    # agent_data[7] = 2.0 * int(node.total_cells - node.total_switch - node.total_switches_neighbors > 0) - 1.0
    # agent_data[8] = node.first_switch_free
    # agent_data[9] = node.first_switch_neighbor
    # agent_data[2] = 2.0 * int(node.num_agents_malfunctioning > 0) - 1.0
    # agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node: TreeObsForRailEnv.Node, dist_min_to_target: int,
                                       current_tree_depth: int,
                                       max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [0] * num_remaining_nodes * 3, [0] * num_remaining_nodes * 1, [0] * num_remaining_nodes * 3

    data, distance, agent_data = _split_node_into_feature_groups(node, dist_min_to_target)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction],
                                                                                    node.dist_min_to_target,
                                                                                    current_tree_depth + 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree: TreeObsForRailEnv.Node, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree, 1000000.0)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction],
                                                                                    1000000.0,
                                                                                    1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation: TreeObsForRailEnv.Node, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)
    # data = norm_obs_clip(data, fixed_radius=observation_radius)
    # print(distance)
    # distance[distance != -np.inf] -= distance[0]
    # distance[distance != -np.inf] = -np.sign(distance[distance != -np.inf])
    # print(distance)
    # distance = norm_obs_clip(distance, normalize_to_range=False)
    # agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    # print(normalized_obs)

    # navigate_info
    navigate_info = np.zeros(4)
    action_info = np.zeros(4)
    np.seterr(all='raise')
    try:
        dm = observation.dist_min_to_target
        if observation.childs['L'] != -np.inf:
            navigate_info[0] = dm - observation.childs['L'].dist_min_to_target
            action_info[0] = 1
        if observation.childs['F'] != -np.inf:
            navigate_info[1] = dm - observation.childs['F'].dist_min_to_target
            action_info[1] = 1
        if observation.childs['R'] != -np.inf:
            navigate_info[2] = dm - observation.childs['R'].dist_min_to_target
            action_info[2] = 1
        if observation.childs['B'] != -np.inf:
            navigate_info[3] = dm - observation.childs['B'].dist_min_to_target
            action_info[3] = 1
    except:
        navigate_info = np.ones(4)
        normalized_obs = np.zeros(len(normalized_obs))

    # navigate_info_2 = np.copy(navigate_info)
    # max_v = np.max(navigate_info_2)
    # navigate_info_2 = navigate_info_2 / max_v
    # navigate_info_2[navigate_info_2 < 1] = -1

    max_v = np.max(navigate_info)

    if max_v == 0:
        max_v = 1.0

    navigate_info = navigate_info / max_v
    navigate_info[navigate_info < 0] = -1


    # navigate_info[abs(navigate_info) < 1] = 0
    # normalized_obs = navigate_info

    # navigate_info = np.concatenate((navigate_info, action_info))
    normalized_obs = np.concatenate((navigate_info, normalized_obs))
    # normalized_obs = np.concatenate((navigate_info, navigate_info_2))
    # print(normalized_obs)

    #print(len(normalized_obs))

    return normalized_obs


class TreeObsForRailEnvRLLibWrapper(ObservationBuilder):

    def __init__(self, tree_obs_builder: TreeObsForRailEnv):
        super().__init__()
        self._builder = tree_obs_builder

    @property
    def observation_dim(self):
        return self._builder.observation_dim

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs = self._builder.get(handle)
        return normalize_observation(obs, self._builder.max_depth, observation_radius=10) if obs is not None else obs

    def get_many(self, handles: Optional[List[int]] = None):
        return {k: normalize_observation(o, self._builder.max_depth, observation_radius=10)
                for k, o in self._builder.get_many(handles).items() if o is not None}

    def util_print_obs_subtree(self, tree):
        self._builder.util_print_obs_subtree(tree)

    def print_subtree(self, node, label, indent):
        self._builder.print_subtree(node, label, indent)

    def set_env(self, env):
        self._builder.set_env(env)

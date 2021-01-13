import numpy as np


tree_explored_actions_char = ['L', 'F', 'R', 'B']

def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(14)
    number = np.zeros(17)
    bol_data = np.zeros(4)
    speed = np.zeros(6)

    if node is None:
        return data, number, bol_data, speed

    data[0] = node.dist_opposite_direction if node.dist_opposite_direction != np.inf and node.dist_opposite_direction != -np.inf else 0
    data[1] = node.potential_conflict if node.potential_conflict != np.inf and node.potential_conflict != -np.inf else 0
    data[2] = node.dist_fastest_opposite_direction if node.dist_fastest_opposite_direction != np.inf and node.dist_fastest_opposite_direction != -np.inf else 0
    data[3] = node.dist_slowest_same_directon if node.dist_slowest_same_directon != np.inf and node.dist_slowest_same_directon != -np.inf else 0
    data[4] = node.malfunctioning_agent if node.malfunctioning_agent != np.inf and node.malfunctioning_agent != -np.inf else 0
    data[5] = node.dist_own_target if node.dist_own_target != np.inf and node.dist_own_target != -np.inf else 0
    data[6] = node.min_dist_others_target if node.min_dist_others_target != np.inf and node.min_dist_others_target != -np.inf else 0
    data[7] = node.dist_same_direction if node.dist_same_direction != np.inf and node.dist_same_direction != -np.inf else 0
    data[8] = node.dist_usable_switch if node.dist_usable_switch != np.inf and node.dist_usable_switch != -np.inf else 0
    data[9] = node.dist_unusable_switch if node.dist_unusable_switch != np.inf and node.dist_unusable_switch != -np.inf else 0
    data[10] = node.mean_agent_distance_to_target_same_direction if node.mean_agent_distance_to_target_same_direction != np.inf and node.mean_agent_distance_to_target_same_direction != -np.inf else 0
    data[11] = node.std_agent_distance_to_target_same_direction if node.std_agent_distance_to_target_same_direction != np.inf and node.std_agent_distance_to_target_same_direction != -np.inf else 0
    data[12] = node.mean_agent_distance_to_target_diff_direction if node.mean_agent_distance_to_target_diff_direction != np.inf and node.mean_agent_distance_to_target_diff_direction != -np.inf else 0
    data[13] = node.std_agent_distance_to_target_diff_direction if node.std_agent_distance_to_target_diff_direction != np.inf and node.std_agent_distance_to_target_diff_direction != -np.inf else 0

    number[0] = node.number_of_slower_agents_same_direction if node.number_of_slower_agents_same_direction != np.inf and node.number_of_slower_agents_same_direction != -np.inf else 0
    number[1] = node.number_of_faster_agents_same_direction if node.number_of_faster_agents_same_direction != np.inf and node.number_of_faster_agents_same_direction != -np.inf else 0
    number[2] = node.number_of_same_speed_agents_same_direction if node.number_of_same_speed_agents_same_direction != np.inf and node.number_of_same_speed_agents_same_direction != -np.inf else 0
    number[3] = node.number_of_slower_agents_opposite_direction if node.number_of_slower_agents_opposite_direction != np.inf and node.number_of_slower_agents_opposite_direction != -np.inf else 0
    number[4] = node.number_of_faster_agents_opposite_direction if node.number_of_faster_agents_opposite_direction != np.inf and node.number_of_faster_agents_opposite_direction != -np.inf else 0
    number[5] = node.number_of_same_speed_agents_opposite_direction if node.number_of_same_speed_agents_opposite_direction != np.inf and node.number_of_same_speed_agents_opposite_direction != -np.inf else 0
    number[6] = node.percentage_active_agents if node.percentage_active_agents != np.inf and node.percentage_active_agents != -np.inf else 0
    number[7] = node.percentage_done_agents if node.percentage_done_agents != np.inf and node.percentage_done_agents != -np.inf else 0
    number[8] = node.percentage_ready_to_depart_agents if node.percentage_ready_to_depart_agents != np.inf and node.percentage_ready_to_depart_agents != -np.inf else 0
    number[9] = node.number_of_usable_switches_on_path if node.number_of_usable_switches_on_path != np.inf and node.number_of_usable_switches_on_path != -np.inf else 0
    number[10] = node.number_of_unusable_switches_on_path if node.number_of_unusable_switches_on_path != np.inf and node.number_of_unusable_switches_on_path != -np.inf else 0
    number[11] = node.sum_priorities_same_direction if node.sum_priorities_same_direction != np.inf and node.sum_priorities_same_direction != -np.inf else 0
    number[12] = node.sum_priorities_diff_direction if node.sum_priorities_diff_direction != np.inf and node.sum_priorities_diff_direction != -np.inf else 0
    number[13] = node.mean_agent_malfunction_same_direction if node.mean_agent_malfunction_same_direction != np.inf and node.mean_agent_malfunction_same_direction != -np.inf else 0
    number[14] = node.std_agent_malfunction_same_direction if node.std_agent_malfunction_same_direction != np.inf and node.std_agent_malfunction_same_direction != -np.inf else 0
    number[15] = node.mean_agent_malfunction_diff_direction if node.mean_agent_malfunction_diff_direction != np.inf and node.mean_agent_malfunction_diff_direction != -np.inf else 0
    number[16] = node.std_agent_malfunction_diff_direction if node.std_agent_malfunction_diff_direction != np.inf and node.std_agent_malfunction_diff_direction != -np.inf else 0

    bol_data[0] = node.priority if node.priority != np.inf and node.priority != -np.inf else 0
    bol_data[1] = node.other_agent_wants_to_occupy_cell_in_path if node.other_agent_wants_to_occupy_cell_in_path != np.inf and node.other_agent_wants_to_occupy_cell_in_path != -np.inf else 0
    bol_data[2] = node.deadlock if node.deadlock != np.inf and node.deadlock != -np.inf else 0
    bol_data[3] = node.currently_on_switch if node.currently_on_switch != np.inf and node.currently_on_switch != -np.inf else 0

    speed[0] = node.fastest_opposite_direction if node.fastest_opposite_direction != np.inf and node.fastest_opposite_direction != -np.inf else 0
    speed[1] = node.slowest_same_dir if node.slowest_same_dir != np.inf and node.slowest_same_dir != -np.inf else 0
    speed[2] = node.mean_agent_speed_same_direction if node.mean_agent_speed_same_direction != np.inf and node.mean_agent_speed_same_direction != -np.inf else 0
    speed[3] = node.std_agent_speed_same_direction if node.std_agent_speed_same_direction != np.inf and node.std_agent_speed_same_direction != -np.inf else 0
    speed[4] = node.mean_agent_speed_diff_direction if node.mean_agent_speed_diff_direction != np.inf and node.mean_agent_speed_diff_direction != -np.inf else 0
    speed[5] = node.std_agent_speed_diff_direction if node.std_agent_speed_diff_direction != np.inf and node.std_agent_speed_diff_direction != -np.inf else 0

    return data, number, bol_data, speed

def split_list_into_feature_groups(list_nodes: list) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    # if len(list_nodes) == 0 or list_nodes[0] is None:
    #     return None

    data, number, bol_data, speed = _split_node_into_feature_groups(list_nodes[0])


    for node in list_nodes[1:]:
        sub_data, sub_number, sub_bol_data, sub_speed = _split_node_into_feature_groups(node)
        data = np.concatenate((data, sub_data))
        number = np.concatenate((number, sub_number))
        bol_data = np.concatenate((bol_data, sub_bol_data))
        speed = np.concatenate((speed, sub_speed))

    return data, number, bol_data, speed


def _split_subtree_into_feature_groups(node, current_tree_depth: int, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):

    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4**(remaining_depth+1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes*6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes*5

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))
    return data, distance, agent_data


def normalize_observation(observation, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs


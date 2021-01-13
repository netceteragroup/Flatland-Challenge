from envs.flatland.observations.segment_graph import Graph


def get_coords(direction):
    if direction == 0:
        return -1, 0
    elif direction == 1:
        return 0, 1
    elif direction == 2:
        return 1, 0
    elif direction == 3:
        return 0, -1


def stop_deadlock_when_unavoidable(timestamp_segment_dict, to_reset, handle, direction, action, action_mask, old_pos):
    # print(obs[agent_id][8])
    dx, dy = get_new_pos_dx_dy(direction, action)
    new_pos = (old_pos[0] + dx, old_pos[1] + dy)
    # print(handle, direction, old_pos, new_pos)
    fr, to = Graph.agents[handle].CurrentNode, Graph.agents[handle].NextNodes
    segments = []
    for node in to:
        segments.append(Graph.graph_global[fr][node]['segment'])
    curr_segment = None
    for segment in segments:
        for x, y, _ in segment:
            if new_pos == (x, y):
                curr_segment = segment
                break
    if curr_segment is None:
        return timestamp_segment_dict, to_reset, action
    curr_segment = frozenset((x, y) for x, y, _ in curr_segment)
    if curr_segment not in timestamp_segment_dict or not timestamp_segment_dict[curr_segment]:
        timestamp_segment_dict[curr_segment] = True
        # print(f"occupied by {handle} segment: {curr_segment}")
        to_reset.append(curr_segment)

    else:
        # print(f"old action was {action}")
        action = pick_new_action(action, action_mask)
        # print(f"new action is {action}")

    return timestamp_segment_dict, to_reset, action


def reset_timestamp_dict(timestamp_segment_dict, to_reset):
    for segment in to_reset:
        # print(f"removing segment {segment}")
        timestamp_segment_dict[segment] = False
    return timestamp_segment_dict


def pick_new_action(old_action, action_mask):
    action_mask[old_action - 1] = 0
    action_mask[3] = 0
    available = [i + 1 for i in range(len(action_mask)) if action_mask[i] == 1]
    if len(available) == 0:
        return old_action
    return available[0]


def get_new_pos_dx_dy(direc, action):
    if direc == 2:
        if action == 1:
            return 0, 1
        if action == 2:
            return 1, 0
        if action == 3:
            return 0, -1
    if direc == 1:
        if action == 1:
            return -1, 0
        if action == 2:
            return 0, 1
        if action == 3:
            return 1, 0
    if direc == 0:
        if action == 1:
            return 0, -1
        if action == 2:
            return -1, 0
        if action == 3:
            return 0, 1
    if direc == 3:
        if action == 1:
            return 1, 0
        if action == 2:
            return 0, -1
        if action == 3:
            return -1, 0

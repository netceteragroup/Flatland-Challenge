
from typing import Dict, Any, Optional, List

import gym
import numpy as np
from collections import defaultdict
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions

from envs.flatland.observations.segment_graph import Graph

from envs.flatland.utils.gym_env import StepOutput


def available_actions(env: RailEnv, agent: EnvAgent, allow_noop=False) -> List[int]:
    if agent.position is None:
        return [0, 1, 0, 1]
    else:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    # some actions are always available:
    available_acts = [0] * len(RailEnvActions)
    available_acts[RailEnvActions.MOVE_FORWARD] = 1
    available_acts[RailEnvActions.STOP_MOVING] = 1
    if allow_noop:
        available_acts[RailEnvActions.DO_NOTHING] = 1
    # check if turn left/right are available:
    for movement in range(4):
        if possible_transitions[movement]:
            if movement == (agent.direction + 1) % 4:
                available_acts[RailEnvActions.MOVE_RIGHT] = 1
            elif movement == (agent.direction - 1) % 4:
                available_acts[RailEnvActions.MOVE_LEFT] = 1

    return available_acts[1:]


def potential_deadlock_action_masking(env: RailEnv, potential_deadlock: List) -> List[int]:
    avaliable_actions = [0, 0, 0, 1]
    avaliable_actions[0] = 0 if potential_deadlock[0] != 1 and potential_deadlock[0] != -1 else 1
    avaliable_actions[1] = 0 if potential_deadlock[1] != 1 and potential_deadlock[1] != -1 else 1
    avaliable_actions[2] = 0 if potential_deadlock[2] != 1 and potential_deadlock[2] != -1 else 1

    return avaliable_actions


def priority_dist_action_masking(dist_ind, priority) -> List[int]:
    available_actions = [0, 0, 0, 0]
    if priority == 0:
        return [0, 0, 0, 1]
    else:
        available_actions[dist_ind] = 1
        return available_actions


class AvailableActionsWrapper(gym.Wrapper):

    def __init__(self, env, allow_noop=False, potential_deadlock_masking=False) -> None:
        super().__init__(env)
        self._allow_noop = allow_noop
        self._potential_deadlock_masking = potential_deadlock_masking
        self.observation_space = gym.spaces.Dict({
            'obs': self.env.observation_space,
            'available_actions': gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)
        })

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, reward, done, info = self.env.step(action_dict)
        return StepOutput(self._transform_obs(obs), reward, done, info)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        return self._transform_obs(self.env.reset(random_seed))

    def _transform_obs(self, obs):
        rail_env = self.unwrapped.rail_env
        if not self._potential_deadlock_masking:
            return {
                agent_id: {
                    'obs': agent_obs,
                    'available_actions': np.asarray(
                        available_actions(rail_env, rail_env.agents[agent_id], self._allow_noop))
                } for agent_id, agent_obs in obs.items()
            }
        else:
            return {
                agent_id: {
                    'obs': agent_obs,
                    'available_actions': np.asarray(
                        priority_dist_action_masking(agent_obs[0], agent_obs[1]))
                } for agent_id, agent_obs in obs.items()
            }


def find_all_cells_where_agent_can_choose(rail_env: RailEnv):
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(rail_env.height):
        for w in range(rail_env.width):
            pos = (w, h)
            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))


class SkipNoChoiceCellsWrapper(gym.Wrapper):

    def __init__(self, env, accumulate_skipped_rewards: bool, discounting: float) -> None:
        super().__init__(env)
        self._switches = None
        self._switches_neighbors = None
        self._decision_cells = None
        self._accumulate_skipped_rewards = accumulate_skipped_rewards
        self._discounting = discounting
        self._skipped_rewards = defaultdict(list)

    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position is None \
               or agent.position == agent.initial_position \
               or agent.position in self._decision_cells

    def _on_switch(self, agent: EnvAgent):
        return agent.position in self._switches

    def _next_to_switch(self, agent: EnvAgent):
        return agent.position in self._switches_neighbors

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        o, r, d, i = {}, {}, {}, {}
        while len(o) == 0:
            obs, reward, done, info = self.env.step(action_dict)
            for agent_id, agent_obs in obs.items():
                if done[agent_id] or self._on_decision_cell(self.unwrapped.rail_env.agents[agent_id]):
                    o[agent_id] = agent_obs
                    r[agent_id] = reward[agent_id]
                    d[agent_id] = done[agent_id]
                    i[agent_id] = info[agent_id]
                    if self._accumulate_skipped_rewards:
                        discounted_skipped_reward = r[agent_id]
                        for skipped_reward in reversed(self._skipped_rewards[agent_id]):
                            discounted_skipped_reward = self._discounting * discounted_skipped_reward + skipped_reward
                        r[agent_id] = discounted_skipped_reward
                        self._skipped_rewards[agent_id] = []
                elif self._accumulate_skipped_rewards:
                    self._skipped_rewards[agent_id].append(reward[agent_id])
            d['__all__'] = done['__all__']
            action_dict = {}
        return StepOutput(o, r, d, i)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        obs = self.env.reset(random_seed)
        self._switches, self._switches_neighbors, self._decision_cells = \
            find_all_cells_where_agent_can_choose(self.unwrapped.rail_env)
        return obs


class RewardWrapperShortestPathObs(gym.Wrapper):
    def __init__(self, env, rewards) -> None:
        super().__init__(env)

        self._finished_reward = rewards['finished_reward']
        self._invalid_action_reward = rewards['invalid_action_reward']
        self._not_finished_reward = rewards['not_finished_reward']
        self._step_reward = rewards['step_reward']
        self._step_shortest_path = rewards['step_shortest_path']
        self._step_second_shortest_path = rewards['step_second_shortest_path']
        self._deadlock_reward = rewards['deadlock_reward']
        self._dont_move_reward = rewards['dont_move_reward']
        self._deadlock_avoidance_reward = rewards['deadlock_avoidance_reward']
        self._stop_on_switch_reward = rewards['stop_on_switch_reward']
        self._stop_potential_deadlock_reward = rewards['stop_potential_deadlock_reward']
        self._deadlock_unusable_switch_avoidance_reward = rewards['deadlock_unusable_switch_avoidance']
        self._priority_reward = rewards['priority_reward']
        self._priority_reward_shortest_path = rewards['priority_reward_shortest_path']
        self._priority_reward_alternative_path = rewards['priority_reward_alternative_path']
        self._priority_penalty = rewards['priority_penalty']
        self._priority_no_path_penalty = rewards['priority_no_path_penalty']

        rail_env: RailEnv = self.unwrapped.rail_env

        self._prev_dist = {agent.handle: [-1, -1] for agent in rail_env.agents}

        self._prev_action_mask = {agent.handle: available_actions(rail_env, agent, False) for agent in rail_env.agents}
        self._prev_pos = {agent.handle: Graph.get_virtual_position(agent.handle) for agent in rail_env.agents}

        self._prev_potential_deadlock = {agent.handle: (0, 0, 0) for agent in rail_env.agents}
        self._prev_on_switch = {agent.handle: 0 for agent in rail_env.agents}


    @staticmethod
    def reward_function(handle, agent_obs, agent_action, agent_done, agent_status, agent_virtual_pos,
                        _prev_potential_deadlock, _prev_dist, _prev_action_mask, _prev_pos, _prev_on_switch,
                        _finished_reward, _invalid_action_reward, _not_finished_reward, _step_reward,
                        _step_shortest_path, _step_second_shortest_path, _deadlock_reward, _dont_move_reward,
                        _deadlock_avoidance_reward, _stop_on_switch_reward, _stop_potential_deadlock_reward,
                        _deadlock_unusable_switch_avoidance_reward, _priority_reward, _priority_reward_shortest_path,
                        _priority_reward_alternative_path, _priority_penalty, _priority_no_path_penalty):
        if agent_done: # done
            if agent_status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                reward = _finished_reward
            elif agent_obs[7] == 1:
                reward = _deadlock_reward
            else:
                reward = _not_finished_reward

        elif agent_obs[7] == 1: # deadlock
            reward = _deadlock_reward

        else:
            potential_deadlock = [agent_obs[19], agent_obs[20], agent_obs[21]]
            available_dirs = sum(1 for d in potential_deadlock if d != -1)
            deadlock_dirs = sum(1 for d in potential_deadlock if d == 1)

            if agent_action == RailEnvActions.STOP_MOVING:
                if agent_obs[30] == 1:
                    reward = _stop_on_switch_reward
                elif agent_obs[36] == 1:
                    #TODO think about this
                    reward = _deadlock_unusable_switch_avoidance_reward * 1 / agent_obs[35] if agent_obs[35] >= 1 else _stop_on_switch_reward
                # elif (deadlock_dirs / available_dirs) == 1. and agent_action == RailEnvActions.STOP_MOVING:
                #     reward = _stop_potential_deadlock_reward * 1/agent_obs[35] if agent_obs[35] >= 1 else _stop_on_switch_reward
                elif agent_obs[39] == 0:
                    reward = _priority_reward
                else:
                    reward = _dont_move_reward

            elif agent_action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_FORWARD]:
                deadlock_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == 1]
                unavaliable_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == -1]
                if _prev_on_switch == 1 and (agent_action not in deadlock_actions and len(deadlock_actions) > 0) and (
                        agent_action not in unavaliable_actions):
                    reward = _deadlock_avoidance_reward

                elif agent_obs[39] == 1:
                    if agent_obs[9] < _prev_dist[0] and agent_obs[9] < 5000:
                        reward = _priority_reward_shortest_path
                    elif agent_obs[9] < _prev_dist[1] < 5000:
                        reward = _priority_reward_alternative_path
                    else:
                        reward = _priority_no_path_penalty
                elif agent_obs[39] == 0:
                    reward = _priority_penalty

                else:
                    reward = _step_reward

            else:
                reward = -1

        return reward

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        rail_env: RailEnv = self.unwrapped.rail_env
        for handle in action_dict:
            action_dict[handle] += 1
        obs, reward, done, info = self.env.step(action_dict)

        o, r, d, i = {}, {}, {}, {}
        for agent_id, agent_obs in obs.items():
            o[agent_id] = obs[agent_id]
            d[agent_id] = done[agent_id]
            i[agent_id] = info[agent_id]
            r[agent_id] = self.reward_function(handle=agent_id,
                                               agent_obs=agent_obs,
                                               agent_action=action_dict[agent_id],
                                               agent_done=done[agent_id],
                                               agent_status=rail_env.agents[agent_id].status,
                                               agent_virtual_pos=Graph.get_virtual_position(agent_id),
                                               _prev_potential_deadlock=self._prev_potential_deadlock[agent_id],
                                               _prev_dist=self._prev_dist[agent_id],
                                               _prev_pos=self._prev_pos[agent_id],
                                               _prev_action_mask=self._prev_action_mask[agent_id],
                                               _prev_on_switch=self._prev_on_switch[agent_id],
                                               _finished_reward=self._finished_reward,
                                               _invalid_action_reward=self._invalid_action_reward,
                                               _not_finished_reward=self._not_finished_reward,
                                               _step_reward=self._step_reward,
                                               _step_shortest_path=self._step_shortest_path,
                                               _step_second_shortest_path=self._step_second_shortest_path,
                                               _deadlock_reward=self._deadlock_reward,
                                               _dont_move_reward=self._dont_move_reward,
                                               _deadlock_avoidance_reward=self._deadlock_avoidance_reward,
                                               _stop_on_switch_reward=self._stop_on_switch_reward,
                                               _stop_potential_deadlock_reward=self._stop_potential_deadlock_reward,
                                               _deadlock_unusable_switch_avoidance_reward=self._deadlock_unusable_switch_avoidance_reward,
                                               _priority_penalty=self._priority_penalty,
                                               _priority_reward=self._priority_reward,
                                               _priority_reward_alternative_path=self._priority_reward_alternative_path,
                                               _priority_reward_shortest_path=self._priority_reward_shortest_path,
                                               _priority_no_path_penalty=self._priority_no_path_penalty
                                               )

            # set prev_states to the length of shortest path if you go L, then F, then R (L,F,R). That corresponds to
            # features 9, 10, 11 in the feature vector
            # print(f"obs: {o}, reward: {r}, prev_dist: {self._prev_dist}")
            self._prev_dist[agent_id] = (agent_obs[9], agent_obs[15])

            self._prev_action_mask[agent_id] = available_actions(rail_env, rail_env.agents[agent_id], False)

            # update potential_deadlock attribute
            self._prev_potential_deadlock[agent_id] = (agent_obs[19], agent_obs[20], agent_obs[21])

            # update prev_pos
            self._prev_pos[agent_id] = Graph.get_virtual_position(agent_id)
            self._prev_on_switch[agent_id] = agent_obs[30]

        d['__all__'] = done['__all__'] or all(d.values())

        return StepOutput(o, r, d, info={agent: {
            'max_episode_steps': int(4 * 2 * (
                    self.rail_env.width + self.rail_env.height + self.rail_env.get_num_agents() / self.num_cities)),
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d[agent] and agent not in self.rail_env.active_agents,
        } for agent in o.keys()})

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        obs = self.env.reset(random_seed=random_seed)

        self._prev_dist = {k: (o[9], o[15]) for k, o in obs.items()}
        return obs


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, rewards) -> None:
        super().__init__(env)
        # self._finished_reward = rewards['finished_reward']
        # self._invalid_action_reward = rewards['invalid_action_reward']
        # self._not_finished_reward = rewards['not_finished_reward']
        # self._step_reward = rewards['step_reward']
        # self._step_shortest_path = rewards['step_shortest_path']
        # self._step_second_shortest_path = rewards['step_second_shortest_path']
        # self._deadlock_reward = rewards['deadlock_reward']
        # self._dont_move_reward = rewards['dont_move_reward']
        # self._deadlock_avoidance_reward = rewards['deadlock_avoidance_reward']
        # self._stop_on_switch_reward = rewards['stop_on_switch_reward']
        # self._stop_potential_deadlock_reward = rewards['stop_potential_deadlock_reward']
        # self._deadlock_unusable_switch_avoidance_reward = rewards['deadlock_unusable_switch_avoidance']
        # self._priority_reward = rewards['priority_reward']
        # self._priority_reward_shortest_path = rewards['priority_reward_shortest_path']
        # self._priority_reward_alternative_path = rewards['priority_reward_alternative_path']
        # self._priority_penalty = rewards['priority_penalty']
        # self._priority_no_path_penalty = rewards['priority_no_path_penalty']
        self._finished_reward = rewards['finished_reward']
        self._deadlock_reward = rewards['deadlock_reward']
        self._step_reward = rewards['step_reward']
        self._deadlock_unusable_switch_avoidance_reward = rewards['deadlock_unusable_switch_avoidance']
        self._stop_priority_depart = rewards['stop_priority_depart']
        self._stop_no_deadlocks_reward = rewards['stop_no_deadlocks_reward']

        rail_env: RailEnv = self.unwrapped.rail_env

        # self._prev_dist = {}

        # self._prev_action_mask = {agent.handle: available_actions(rail_env, agent, False) for agent in rail_env.agents}
        # self._prev_pos = {agent.handle: Graph.get_virtual_position(agent.handle) for agent in rail_env.agents}
        #
        # self._prev_potential_deadlock = {}
        # self._prev_on_switch = {}
        # self._prev_deadlock_unusable = {}
        self._prev_shortest_action = {}
        self._prev_priority = {}

    def reward_function(self, handle, agent_obs, agent_action, agent_done, agent_status):

        if agent_done:
            if agent_status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                reward = self._finished_reward
            else:
                reward = self._step_reward
        elif agent_obs[5] == 1 and agent_action == RailEnvActions.STOP_MOVING:
            reward = 0
        elif agent_obs[5] == 1 and agent_action != RailEnvActions.STOP_MOVING:
            reward = -10
        else:
            if self._prev_priority[handle] == 0:
                if agent_action == RailEnvActions.STOP_MOVING:
                    reward = 0
                else:
                    reward = -10
            else:
                #if (agent_action - 1) == np.argmax(self._prev_shortest_action[handle]):
                if agent_action != RailEnvActions.STOP_MOVING:
                    reward = 0
                else:
                    reward = -10
            # reward = (1 / agent_obs[4] + self._step_reward) * 0.70
            # if agent_action == RailEnvActions.STOP_MOVING:
            #     if self._prev_priority[handle] == 0 and agent_status == RailAgentStatus.READY_TO_DEPART:
            #         reward = self._stop_priority_depart
            #     elif self._prev_deadlock_unusable[handle] == 1:
            #         reward = self._deadlock_unusable_switch_avoidance_reward
            #
            #     elif 1 not in self._prev_potential_deadlock[handle] and self._prev_deadlock_unusable[handle] == 0 and self._prev_priority[handle] == 1:
            #         reward = self._stop_no_deadlocks_reward

        # if agent_done: # done
        #     if agent_status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
        #         reward = _finished_reward
        #     elif agent_obs[7] == 1:
        #         reward = _deadlock_reward
        #     else:
        #         reward = _not_finished_reward
        #
        # elif agent_obs[7] == 1: # deadlock
        #     reward = _deadlock_reward
        #
        # else:
        #     potential_deadlock = [agent_obs[19], agent_obs[20], agent_obs[21]]
        #     available_dirs = sum(1 for d in potential_deadlock if d != -1)
        #     deadlock_dirs = sum(1 for d in potential_deadlock if d == 1)
        #
        #     if agent_action == RailEnvActions.STOP_MOVING:
        #         if agent_obs[30] == 1:
        #             reward = _stop_on_switch_reward
        #         elif agent_obs[36] == 1:
        #             # TODO think about this
        #             if agent_obs[35] == 1:
        #                 reward = _deadlock_unusable_switch_avoidance_reward
        #             else:
        #                 r = -_deadlock_unusable_switch_avoidance_reward
        #                 reward = -(r**(1/agent_obs[35])*0.5)
        #         # elif (deadlock_dirs / available_dirs) == 1. and agent_action == RailEnvActions.STOP_MOVING:
        #         #     reward = _stop_potential_deadlock_reward * 1/agent_obs[35] if agent_obs[35] >= 1 else _stop_on_switch_reward
        #         elif agent_obs[39] == 0:
        #             reward = _priority_reward
        #         else:
        #             reward = _dont_move_reward
        #
        #     elif agent_action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_FORWARD]:
        #         deadlock_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == 1]
        #         unavaliable_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == -1]
        #         if _prev_on_switch == 1 and (agent_action not in deadlock_actions and len(deadlock_actions) > 0) and (
        #                     agent_action not in unavaliable_actions):
        #             reward = _deadlock_avoidance_reward
        #
        #         elif agent_obs[39] == 1:
        #             if agent_obs[9] < _prev_dist[0] and agent_obs[9] < 5000:
        #                 reward = _priority_reward_shortest_path
        #             elif agent_obs[9] < _prev_dist[1] < 5000:
        #                 reward = _priority_reward_alternative_path
        #             else:
        #                 reward = _priority_no_path_penalty
        #         elif agent_obs[39] == 0:
        #             reward = _priority_penalty
        #
        #         else:
        #             reward = _step_reward
        #
        #     else:
        #         reward = -1
        #
        # return reward
        #

        # if agent_done:
        #     if agent_status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
        #         # agent is done and really done -> give finished reward
        #         reward = _finished_reward
        #     else:
        #         # agent is done but not really done -> give not_finished reward
        #         if agent_obs[7] == 1:
        #             reward = _deadlock_reward
        #         else:
        #             reward = _not_finished_reward
        #
        # elif agent_obs[7] == 1:
        #     reward = _deadlock_reward
        #
        # else:
        #     if agent_obs[9] < _prev_dist[0] and agent_obs[9] != -1:
        #         reward = _step_shortest_path
        #
        #     elif agent_obs[15] < _prev_dist[1] and agent_obs[15] != -1:
        #         reward = _step_second_shortest_path
        #
        #     else:
        #         reward = _step_reward
        #
        #
        #
        #     # invalid action reward
        #     if _prev_action_mask[agent_action-1] == 0:
        #         reward += _invalid_action_reward
        #
        #         # if agent not moving
        #     if tuple(_prev_pos) == tuple(agent_virtual_pos):
        #         reward += _dont_move_reward
        #
        #         # stop on switch
        #     if agent_obs[30] == 1 and agent_action == RailEnvActions.STOP_MOVING:
        #         reward += _stop_on_switch_reward
        #
        #     potential_deadlock = [agent_obs[19], agent_obs[20], agent_obs[21]]
        #     available_dirs = sum(1 for d in potential_deadlock if d != -1)
        #     deadlock_dirs = sum(1 for d in potential_deadlock if d == 1)
        #     if (deadlock_dirs / available_dirs) == 1. and agent_action == RailEnvActions.STOP_MOVING:
        #         reward += _stop_potential_deadlock_reward * 1/agent_obs[35] if agent_obs[35] >= 1 else 0
        #
        #
        #     if agent_obs[36] == 1 and agent_action == RailEnvActions.STOP_MOVING:
        #         reward += _deadlock_unusable_switch_avoidance_reward * 1 / agent_obs[35] if agent_obs[35] >= 1 else 0
        #
        #     # reward if agent avoided deadlock
        #     if _prev_on_switch == 1:
        #         deadlock_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == 1]
        #         unavaliable_actions = [idx + 1 for idx, action in enumerate(_prev_potential_deadlock) if action == -1]
        #         if (agent_action not in deadlock_actions and len(deadlock_actions) > 0) and (
        #                 agent_action not in unavaliable_actions) and (agent_action != RailEnvActions.DO_NOTHING) \
        #                 and (agent_action != RailEnvActions.STOP_MOVING):
        #             reward = _deadlock_avoidance_reward

        return reward

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        rail_env: RailEnv = self.unwrapped.rail_env
        for handle in action_dict:
            action_dict[handle] += 1
            if action_dict[handle] < 4:
                action_dict[handle] = possible_actions_sorted_by_distance(rail_env, handle)[0][0]

        obs, reward, done, info = self.env.step(action_dict)

        o, r, d, i = {}, {}, {}, {}
        for agent_id, agent_obs in obs.items():
            o[agent_id] = obs[agent_id]
            d[agent_id] = done[agent_id]
            i[agent_id] = info[agent_id]
            r[agent_id] = self.reward_function(handle=agent_id,
                                               agent_obs=agent_obs,
                                               agent_action=action_dict[agent_id],
                                               agent_done=done[agent_id],
                                               agent_status=rail_env.agents[agent_id].status,
                                               )

            # print(f"Agent {agent_id}, obs: {agent_obs}, prev_priority: {self._prev_priority[agent_id]}, prev_dist_action: {self._prev_shortest_action[agent_id]}, reward: {r[agent_id]}, action: {action_dict[agent_id] - 1}")
            # set prev_states to the length of shortest path if you go L, then F, then R (L,F,R). That corresponds to
            # features 9, 10, 11 in the feature vector
            # print(f"obs: {o}, reward: {r}, prev_dist: {self._prev_dist}")
            # self._prev_dist[agent_id] = (agent_obs[9], agent_obs[15])

            # self._prev_action_mask[agent_id] = available_actions(rail_env, rail_env.agents[agent_id], False)

            # update potential_deadlock attribute
            # self._prev_potential_deadlock[agent_id] = (agent_obs[10], agent_obs[11], agent_obs[12])

            # update prev_pos
            # self._prev_pos[agent_id] = Graph.get_virtual_position(agent_id)

            # self._prev_on_switch[agent_id] = agent_obs[13]
            self._prev_shortest_action[agent_id] = [agent_obs[0], agent_obs[1], agent_obs[2]]
            self._prev_priority[agent_id] = agent_obs[3]

            # self._prev_deadlock_unusable[agent_id] = agent_obs[19]

        d['__all__'] = done['__all__'] or all(d.values())

        return StepOutput(o, r, d, info={agent: {
            'max_episode_steps': int(4 * 2 * (
                        self.rail_env.width + self.rail_env.height + self.rail_env.get_num_agents() / self.num_cities)),
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d[agent] and agent not in self.rail_env.active_agents,
        } for agent in o.keys()})

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        obs = self.env.reset(random_seed=random_seed)

        # self._prev_dist = {k: (o[9], o[15]) for k, o in obs.items()}

        # self._prev_potential_deadlock = {k: (o[10], o[11], o[12]) for k, o in obs.items()}
        # self._prev_on_switch = {k: o[13] for k, o in obs.items()}
        self._prev_shortest_action = {k: [o[0], o[1], o[2]] for k, o in obs.items()}
        self._prev_priority = {k: o[3] for k, o in obs.items()}
        # self._prev_deadlock_unusable = {k: o[19] for k, o in obs.items()}

        return obs


class SparseRewardWrapper(gym.Wrapper):

    def __init__(self, env, finished_reward=1, not_finished_reward=-1) -> None:
        super().__init__(env)
        self._finished_reward = finished_reward
        self._not_finished_reward = not_finished_reward

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        rail_env: RailEnv = self.unwrapped.rail_env
        for handle in action_dict:
            action_dict[handle] += 1

        obs, reward, done, info = self.env.step(action_dict)

        o, r, d, i = {}, {}, {}, {}
        for agent_id, agent_obs in obs.items():
            o[agent_id] = obs[agent_id]
            d[agent_id] = done[agent_id]
            i[agent_id] = info[agent_id]
            if done[agent_id]:
                if rail_env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    # agent is done and really done -> give finished reward
                    r[agent_id] = self._finished_reward
                else:
                    # agent is done but not really done -> give not_finished reward
                    r[agent_id] = self._not_finished_reward
            else:
                r[agent_id] = 0
        d['__all__'] = done['__all__'] or all(d.values())

        return StepOutput(o, r, d, i)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        return self.env.reset(random_seed)


class DeadlockWrapper(gym.Wrapper):

    def __init__(self, env, deadlock_reward=-1) -> None:
        super().__init__(env)
        self._deadlock_reward = deadlock_reward
        self._deadlocked_agents = []

    def check_deadlock(self):  # -> Set[int]:
        rail_env: RailEnv = self.unwrapped.rail_env
        new_deadlocked_agents = []
        for agent in rail_env.agents:
            if agent.status == RailAgentStatus.ACTIVE and agent.handle not in self._deadlocked_agents:
                position = agent.position
                direction = agent.direction
                while position is not None:
                    possible_transitions = rail_env.rail.get_transitions(*position, direction)
                    num_transitions = np.count_nonzero(possible_transitions)
                    if num_transitions == 1:
                        new_direction_me = np.argmax(possible_transitions)
                        new_cell_me = get_new_position(position, new_direction_me)
                        opp_agent = rail_env.agent_positions[new_cell_me]
                        if opp_agent != -1:
                            opp_position = rail_env.agents[opp_agent].position
                            opp_direction = rail_env.agents[opp_agent].direction
                            opp_possible_transitions = rail_env.rail.get_transitions(*opp_position, opp_direction)
                            opp_num_transitions = np.count_nonzero(opp_possible_transitions)
                            if opp_num_transitions == 1:
                                if opp_direction != direction:
                                    self._deadlocked_agents.append(agent.handle)
                                    new_deadlocked_agents.append(agent.handle)
                                    position = None
                                else:
                                    position = new_cell_me
                                    direction = new_direction_me
                            else:
                                position = new_cell_me
                                direction = new_direction_me
                        else:
                            position = None
                    else:
                        position = None
        return new_deadlocked_agents

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, reward, done, info = self.env.step(action_dict)

        if self._deadlock_reward != 0:
            new_deadlocked_agents = self.check_deadlock()
        else:
            new_deadlocked_agents = []

        o, r, d, i = {}, {}, {}, {}
        for agent_id, agent_obs in obs.items():
            if agent_id not in self._deadlocked_agents or agent_id in new_deadlocked_agents:
                o[agent_id] = obs[agent_id]
                d[agent_id] = done[agent_id]
                i[agent_id] = info[agent_id]
                r[agent_id] = reward[agent_id]
                if agent_id in new_deadlocked_agents:
                    # agent is in deadlocked (and was not before) -> give deadlock reward and set to done
                    r[agent_id] += self._deadlock_reward
                    d[agent_id] = True
        d['__all__'] = done['__all__'] or all(d.values())

        return StepOutput(o, r, d, i)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._deadlocked_agents = []
        return self.env.reset(random_seed)


def possible_actions_sorted_by_distance(env: RailEnv, handle: int):
    agent = env.agents[handle]

    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target
    else:
        return None

    possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
    distance_map = env.distance_map.get()[handle]
    possible_steps = []
    for movement in list(range(4)):
        if possible_transitions[movement]:
            if movement == agent.direction:
                action = RailEnvActions.MOVE_FORWARD
            elif movement == (agent.direction + 1) % 4:
                action = RailEnvActions.MOVE_RIGHT
            elif movement == (agent.direction - 1) % 4:
                action = RailEnvActions.MOVE_LEFT
            else:
                raise ValueError("Wtf, debug this shit.")
            distance = distance_map[get_new_position(agent_virtual_position, movement) + (movement,)]
            possible_steps.append((action, distance))
    possible_steps = sorted(possible_steps, key=lambda step: step[1])

    if len(possible_steps) == 1:
        return possible_steps * 2
    else:
        return possible_steps


class ShortestPathActionWrapper(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        print("Apply ShortestPathActionWrapper")
        self.action_space = gym.spaces.Discrete(n=3)  # stop, shortest path, other direction

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        rail_env: RailEnv = self.env.unwrapped.rail_env
        transformed_action_dict = {}
        for agent_id, action in action_dict.items():
            if action == 0:
                transformed_action_dict[agent_id] = action
            else:
                assert action in [1, 2]
                transformed_action_dict[agent_id] = possible_actions_sorted_by_distance(rail_env, agent_id)[action - 1][
                    0]
        step_output = self.env.step(transformed_action_dict)
        return step_output

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        return self.env.reset(random_seed)


class DeadlockResolutionWrapper(gym.Wrapper):

    def __init__(self, env, deadlock_reward=0) -> None:
        super().__init__(env)
        self._deadlock_reward = deadlock_reward
        self._num_swaps = defaultdict(int)

    def get_deadlocks(self, agent: EnvAgent, seen: List[int]) -> EnvAgent:
        # abort if agent already checked
        if agent.handle in seen:
            # handle circular deadlock
            seen.append(agent.handle)
            # return
            return []
        # add agent to seen agents
        seen.append(agent.handle)

        # get rail environment
        rail_env: RailEnv = self.unwrapped.rail_env
        # get transitions for agent's position and direction
        transitions = rail_env.rail.get_transitions(*agent.position, agent.direction)
        num_possible_transitions = np.count_nonzero(transitions)
        # initialize list to assign deadlocked agents to directions
        deadlocked_agents = [None] * len(transitions)
        # check if all possible transitions are blocked
        for direction, transition in enumerate(transitions):
            # only check transitions > 0 but iterate through all to get direction
            if transition > 0:
                # get opposite agent in direction of travel if cell is occuppied
                new_position = get_new_position(agent.position, direction)
                i_opp_agent = rail_env.agent_positions[new_position]
                if i_opp_agent != -1:
                    opp_agent = rail_env.agents[i_opp_agent]
                    # get blocking agents of opposite agent
                    blocking_agents = self.get_deadlocks(opp_agent, seen)
                    # add opposite agent to deadlocked agents if blocked by
                    # checking agent. also add opposite agent if it is part
                    # of a circular blocking structure.
                    if agent in blocking_agents or seen[0] == seen[-1]:
                        deadlocked_agents[direction] = opp_agent

        # return deadlocked agents if applicable
        num_deadlocked_agents = np.count_nonzero(deadlocked_agents)
        if num_deadlocked_agents > 0:
            # deadlock has to be resolved only if no transition is possible
            if num_deadlocked_agents == num_possible_transitions:
                return deadlocked_agents
            # workaround for already commited agent inside cell that is blocked by at least one agent
            if agent.speed_data['position_fraction'] > 1:
                return deadlocked_agents

        return []

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, reward, done, info = self.env.step(action_dict)
        # get rail environment
        rail_env: RailEnv = self.unwrapped.rail_env
        # check agents that have status ACTIVE for deadlocks, env.active_agents contains also other agents
        active_agents = [agent for agent in rail_env.agents if agent.status == RailAgentStatus.ACTIVE]
        for agent in active_agents:
            deadlocked_agents = self.get_deadlocks(agent, [])
            if len(deadlocked_agents) > 0:
                # favor transition in front as most natural
                d_agent = deadlocked_agents[agent.direction]
                # get most likely transition if straight forward is no valid transition
                if d_agent is None:
                    transitions = rail_env.rail.get_transitions(*agent.position, agent.direction)
                    agent.direction = np.argmax(transitions)
                    d_agent = deadlocked_agents[agent.direction]
                # already commited agent can have only one transition blocked
                if d_agent is None:
                    d_agent = [a for a in deadlocked_agents if a is not None][0]
                # swap the deadlocked pair
                agent.position, d_agent.position = d_agent.position, agent.position
                rail_env.agent_positions[agent.position] = agent.handle
                rail_env.agent_positions[d_agent.position] = d_agent.handle
                # set direction of blocking agent because of corners
                d_agent.direction = (agent.direction + 2) % 4
                # position is exact after swap
                agent.speed_data['position_fraction'] = 0.0
                d_agent.speed_data['position_fraction'] = 0.0
                # punish agents for deadlock
                reward[agent.handle] += self._deadlock_reward
                reward[d_agent.handle] += self._deadlock_reward
                # increase swap counter in info dict
                self._num_swaps[agent.handle] += 1
                self._num_swaps[d_agent.handle] += 1

        for i_agent in info:
            info[i_agent]['num_swaps'] = self._num_swaps[i_agent]

        return obs, reward, done, info

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._num_swaps = defaultdict(int)
        return self.env.reset(random_seed)


class FlatlandRenderWrapper(RailEnv, gym.Env):

    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    # # Set these in ALL subclasses
    # observation_space = None

    def __init__(self, use_renderer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_renderer = use_renderer
        self.renderer = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 10,
            'semantics.autoreset': True
        }
        if self.use_renderer:
            self.initialize_renderer()

    def reset(self, *args, **kwargs):
        if self.use_renderer:
            if self.renderer:  # TODO: Errors with RLLib with renderer as None.
                self.renderer.reset()
        return super().reset(*args, **kwargs)

    def render(self, mode='human'):
        """
        This methods provides the option to render the
        environment's behavior to a window which should be
        readable to the human eye if mode is set to 'human'.
        """
        if not self.use_renderer:
            return

        if not self.renderer:
            self.initialize_renderer(mode=mode)

        return self.update_renderer(mode=mode)

    def initialize_renderer(self, mode="human"):
        # Initiate the renderer
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        self.renderer = RenderTool(self, gl="PGL",  # gl="TKPILSVG",
                                   agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                   show_debug=False,
                                   screen_height=600,  # Adjust these parameters to fit your resolution
                                   screen_width=800)  # Adjust these parameters to fit your resolution

    def update_renderer(self, mode='human'):
        image = self.renderer.render_env(show=True, show_observations=False, show_predictions=False,
                                         return_image=True)
        return image[:, :, :3]

    def set_renderer(self, renderer):
        self.use_renderer = renderer
        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

    def close(self):
        super().close()
        if self.renderer:
            try:
                self.renderer.close_window()
                self.renderer = None
            except Exception as e:
                # This is since the last step(Due to a stopping criteria) is skipped by rllib
                # Due to this done is not true and the env does not close
                # Finally the env is closed when RLLib exits but at that time there is no window
                # and hence the error
                print("Could Not close window due to:", e)

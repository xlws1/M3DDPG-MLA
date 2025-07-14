import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union

class Multiagent_wrapper(ABC, gym.Wrapper):
    def __init__(self, env: gym.Env, state_space: gym.Space, num_agents: int, action_spaces: List[gym.Space], observation_spaces: List[gym.Space]) -> None:
        """initalizes Multiagent wrapper

        Args:
            env (gym.Env): underlying gym environment
            state_space (gym.Space): Observation space of the underlying environment
            num_agents (int): number of agents participating in this environment
            action_spaces (List[gym.Space]): List of action spaces of agents. According to order of agents.
            observation_spaces (List[gym.Space]): List of observation spaces of agents. Accoding to order of agents.
        """
        gym.Wrapper.__init__(self, env)

        self.state_space = state_space
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

    def step(self, actions: List[np.array]) -> Tuple[np.array, List[np.array], List[float], bool, Dict]:
        """
        执行一个环境步骤

        Args:
            actions (List[np.array]): 每个智能体的动作

        Returns:
            Tuple[np.array, List[np.array], List[float], bool, Dict]:
            - 下一个状态
            - 每个智能体的观测值
            - 每个智能体的奖励
            - 是否结束
            - 额外信息
        """
        # 组合动作
        joint_action = self._build_joint_action(actions)

        # 执行环境步骤
        step_result = self.env.step(joint_action)

        # 处理不同长度的返回值
        if len(step_result) == 5:
            next_state, reward, done, truncated, info = step_result
            done = done or truncated  # 结合 done 和 truncated
        elif len(step_result) == 4:
            next_state, reward, done, info = step_result
        else:
            raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")

        # 构建观测值
        next_observations = self._build_observations(next_state)

        # 构建奖励
        rewards = self._build_rewards(next_state, reward, info)

        return next_state, next_observations, rewards, done, info

    def reset(self):
        # 重置底层环境
        state = self.env.reset()

        # 如果 state 是元组，取第一个元素
        if isinstance(state, tuple):
            state = state[0]

        # 构建观测值
        observations = self._build_observations(state)

        return state, observations

    def _build_state(self, state: np.array) -> np.array:
        """This function is optional, it can be used to preprocess the state

        Args:
            state (np.array): raw state given by the original environment

        Returns:
            np.array: preprocessed state
        """
        return state

    @abstractmethod
    def _build_joint_action(self, actions: List[np.array]) -> np.array:
        """This function needs to merge the actions together such that the underlying environment can process them.

        Args:
            actions (List[numpy.array]): List of actions according to the order of agents.

        Returns:
            numpy.array: merged actions such that the underlying environment can process them.
        """
        pass

    @abstractmethod
    def _build_observations(self, state: np.array) -> List[np.array]:
        """This function needs to split up the state into the observations of the single agents.

        Args:
            state (numpy.array): total state of the environment

        Returns:
            List[numpy.array]: List of observations according to order of agents.
        """
        pass

    @abstractmethod
    def _build_rewards(self, state: np.array, reward: float, info: Union[None, Dict]) -> List[float]:
        """This function needs to determin the rewards for all agents

        Args:
            state (numpy.array): Total state of the underlying environment
            reward (float): reward of the underlying environment
            info (Union[None, Dict])): info of the underlying environment

        Returns:
            List[float]: List of rewards according to order of agents
        """
        pass
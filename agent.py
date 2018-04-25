from abc import ABCMeta, abstractmethod
import numpy as np
import gym
class BaseAgent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, config, env, agent):
        envs = config['envs']
        self.env_name = env
        env = envs[env]
        self.env = gym.make(env['name'])

        self.win_condition_score = env['win_condition']['score']
        self.win_condition_over = env['win_condition']['over']

        self.num_actions = env['num_actions']
        self.state_size = env['state_space']

        self.max_episodes = env['max_episodes']
        self.total_rewards = []
        self.eps_reward = 0
        self.episode = -1
        self.won = False

        self.current_state = None
        self.done = False

        self.agent_config = config['algorithms'][agent]

    def get_attribute(self, attribute):
        if self.env_name in self.agent_config and attribute in self.agent_config[self.env_name]:
            return self.agent_config[self.env_name][attribute]
        return self.agent_config[attribute]

    @abstractmethod
    def create_network(self):
        raise NotImplementedError

    @abstractmethod
    def loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, render=False):
        raise NotImplementedError

    def reset(self):
        self.current_state = self.env.reset()
        self.done = False
        self.eps_reward = 0
        self.episode += 1

    @abstractmethod
    def run(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, current_state):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, current_state, action, reward, done, next_state):
        raise NotImplementedError

    @abstractmethod
    def get(self):
        raise NotImplementedError

    def one_hot_encode(self, action):
        actions = np.zeros(self.num_actions)
        actions[action] = 1
        return actions

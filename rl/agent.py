from abc import ABCMeta, abstractmethod


class BaseAgent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, env):
        self.env = env
        # TODO: Get the action regardless of the env
        self.num_actions = self.env.action_space.n
        self.state_size = len(self.env.observation_space.high)
        self.eps_reward = 0

        self.current_state = None
        self.done = False

    @abstractmethod
    def create_network(self):
        raise NotImplementedError

    @abstractmethod
    def loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, render):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

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

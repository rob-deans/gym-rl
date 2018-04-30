from agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, config, env):
        super(RandomAgent, self).__init__(config, env, 'random')

    def create_network(self):
        pass

    def loss_fn(self):
        pass

    def step(self, render=False):
        if render:
            self.env.render()

        action = self.get_action(self.current_state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        self.add(self.current_state, action, reward, done, next_state)

        self.current_state = next_state

        return reward, done

    def train(self):
        pass

    def run(self, state):
        pass

    def get_action(self, current_state):
        return self.env.action_space.sample()

    def add(self, current_state, action, reward, done, next_state):
        pass

    def get(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def __str__(self):
        return 'random'

import numpy as np


class Logger:
    def __init__(self):
        self.rewards = []
        self.episode = 0
        self.epsilon = 1.
        self.average_num = 100
        self.log_every = 1
        self.log_avg_every = 30

    def _log_episode(self):
        print('Episode: {} | Reward: {}'.format(self.episode, self.rewards[-1]))

    def _log_average(self):
        print
        print('Stats between episode {} and {}'.format(self.episode - self.average_num, self.episode))
        r = self.rewards[-self.average_num:]
        print('Average Reward: {} | Max: {} | Min: {}'.format(np.mean(r), max(r), min(r)))
        print

    def log(self, reward):
        self.rewards.append(reward)
        self.episode = len(self.rewards)

        if self.episode % self.log_every != 0:
            return

        if self.episode % self.log_avg_every == 0 and self.episode >= self.average_num:
            self._log_average()
        else:
            self._log_episode()

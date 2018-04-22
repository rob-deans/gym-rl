import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from scipy.interpolate import spline


class Statistics:
    def __init__(self, config):
        self.types = []
        self.envs = []
        self.rewards = []

        e = config['test']['envs']
        self.average_rewards = {}
        self.best = {}
        self.worst = {}

        for i in e:
            self.average_rewards[i] = {}
            self.best[i] = {}
            self.worst[i] = {}

        self.time_taken = []

    def add(self, agent, env):
        self.types.append(str(agent))
        self.envs.append(env)
        self.rewards.append(agent.total_rewards)

        r = []
        for i, t in enumerate(self.types):
            if str(agent) == t and env == self.envs[i]:
                r.append(self.rewards[i])
        b = r[0]
        w = r[0]
        for run in r:
            if np.mean(run) > np.mean(b):
                b = run
            if np.mean(run) < np.mean(b):
                w = run

        self.average_rewards[env][str(agent)] = np.mean(r, axis=0)
        # self.average_rewards[str(agent)] = np.mean(r, axis=0)
        self.best[env][str(agent)] = b
        self.worst[env][str(agent)] = w

    def save(self):
        pickle.dump(self, open('stats_ac_lunar.pkl', 'wb'))
        # pass

    def visualise(self, load=False):
        if load:
            pickle.load(open('stats.pkl', 'rb'))
            self.average_rewards = self.process()
        envs = self.average_rewards.keys()
        for e in envs:
            types = self.average_rewards[e].keys()
            self.process()
            for t in types:
                agent = str(t)
                plt.title('{} | {}'.format(e, agent))
                plt.plot(self.average_rewards[e][agent], label='Average')
                plt.plot(self.best[e][agent], label='Best')
                plt.plot(self.worst[e][agent], label='Worst')
                plt.ylabel('Episode Reward')
                plt.xlabel('Episode')
                plt.legend()
                plt.savefig('{}_{}'.format(e, agent))
                plt.clf()

    def process(self):
        df = pd.Series(len(self.average_rewards))
        ma_counts = df.rolling(window=10).mean()
        ma_counts = ma_counts.values
        cleaned_list = [x for x in ma_counts if str(x) != 'nan']
        cleaned_list = np.asarray(cleaned_list)
        # 300 represents number of points to make between T.min and T.max
        x_new = np.linspace(cleaned_list.min(), cleaned_list.max(), len(cleaned_list))
        episodes = np.arange(0, len(cleaned_list))
        return spline(episodes, cleaned_list, x_new)

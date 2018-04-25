import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd


class Statistics:
    def __init__(self, config):
        self.types = []
        self.envs = []
        self.rewards = []

        e = config['test']['envs'] if config is not None else []
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
        pickle.dump(self, open('policy_cartpole.pkl', 'wb'))
        # pass

    def visualise(self):
        envs = self.average_rewards.keys()
        for e in envs:
            types = self.average_rewards[e].keys()
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

    def smooth_vis(self, f):
        self = pickle.load(open(f, 'rb'))
        envs = self.average_rewards.keys()

        for e in envs:
            types = self.average_rewards[e].keys()
            # self.process()
            for t in types:
                agent = str(t)
                plt.title('{} | {}'.format(e, agent))
                avg = self.process(e, agent, self.average_rewards)
                best = self.process(e, agent, self.best)
                worst = self.process(e, agent, self.worst)
                plt.plot(avg, label='Average')
                plt.plot(best, label='Best')
                plt.plot(worst, label='Worst')
                plt.ylabel('Episode Reward')
                plt.xlabel('Episode')
                plt.legend()
                # plt.savefig('{}_{}'.format(e, agent))
                # plt.clf()
                plt.plot()
                plt.show()

    def process(self, env, agent, t):
        x = t[env][agent]
        df = pd.Series(x)
        ma_counts = df.rolling(window=10).mean()
        ma_counts = ma_counts.values
        cleaned_list = [x for x in ma_counts if str(x) != 'nan']
        cleaned_list = np.asarray(cleaned_list)
        return cleaned_list

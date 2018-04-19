import numpy as np
from matplotlib import pyplot as plt


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
        pass

    def visualise(self, animate=False):
        # fig = plt.figure()
        #
        # ax = fig.add_subplot(111)
        #
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['right'].set_color('none')
        # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

        # types = self.average_rewards.keys()
        envs = self.average_rewards.keys()
        for e in envs:
            types = self.average_rewards[e].keys()
            for t in types:
                agent = str(t)
                # ax1 = fig.add_subplot(len(types), 1, i+1)
                # ax1.set_title(str(t))
                # ax1.plot(self.average_rewards[temp], label='Average')
                # ax1.plot(self.best[temp], label='Best')
                # ax1.plot(self.worst[temp], label='Worst')
                # ax1.legend()
                plt.title('{} | {}'.format(e, agent))
                plt.plot(self.average_rewards[e][agent], label='Average')
                plt.plot(self.best[e][agent], label='Best')
                plt.plot(self.worst[e][agent], label='Worst')
                plt.ylabel('Reward')
                plt.xlabel('Episode')
                plt.legend()
                plt.savefig('{}_{}'.format(e, agent))
                plt.clf()
                # ax1.legend()

        # for i in range(len(self.average_rewards)):
        #     ax1 = fig.add_subplot(211)
        #     ax1.set_title('DeepQ')
        #     ax1.plot(self.average_rewards['deepq'], label='Average')
        #     ax1.plot(self.best['deepq'], label='Best')
        #     ax1.plot(self.worst['deepq'], label='Worst')
        #     ax1.legend()
        #
        # ax2 = fig.add_subplot(212)
        # ax2.set_title('Policy Gradient')
        # ax2.plot(self.average_rewards['policy'], label='Average')
        # ax2.plot(self.best['policy'], label='Best')
        # ax2.plot(self.worst['policy'], label='Worst')
        # ax2.legend()

        # ax.set_xlabel('Episode')
        # ax.set_ylabel('Reward')
        #
        # plt.show()

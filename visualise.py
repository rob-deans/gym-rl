from statistics import Statistics


stats = Statistics(None)

stats.smooth_vis_sc2('random_results.pkl')
# import pickle
# import matplotlib.pyplot as plt


# font = {'size': 12}
#
# plt.rc('font', **font)
#
# q_vals = pickle.load(open('q_vals_dqn.pkl', 'rb'))
# plt.plot(q_vals)
# plt.ylabel('Q-value')
# plt.xlabel('Timestep')
# plt.show()
#
#
# rewards = pickle.load(open('random_random.pkl', 'rb'))
# print(rewards)
#
# envs = rewards.average_rewards.keys()
#
# for e in envs:
#     types = rewards.average_rewards[e].keys()
#     # self.process()
#     for t in types:
#         agent = str(t)
#         x = rewards.best[e][agent]
#         print(agent, e, np.max(x), 'best max')
#         x = np.mean(rewards.best[e][agent])
#         print(agent, e, x, 'best average')
#         y = np.mean(rewards.worst[e][agent])
#         print(agent, e, y, 'worst average')
#         print(agent, e, math.fabs(x-y), 'difference')

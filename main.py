import gym
from rl import DeepQAgent
from rl import PolicyGradient
from rl import ActorCritic
import logger


def step(agent, render=False):
    done = False
    agent.reset()

    while not done:

        reward, done = agent.step()
        agent.eps_reward += reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = ActorCritic(env)
    logger = logger.Logger()

    for _ in range(agent.max_episodes):
        step(agent)
        # if agent.epsilon > agent.epsilon_end:
        #     agent.epsilon *= agent.epsilon_decay
        logger.log(agent.eps_reward)

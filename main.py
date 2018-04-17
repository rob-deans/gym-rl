import gym
from rl import DeepQAgent
from rl import PolicyGradient
import logger


def step(agent, render=False):
    current_state = env.reset()
    done = False
    eps_reward = 0
    while not done:

        if render:
            env.render()

        action = agent.get_action(current_state)

        next_state, reward, done, _ = env.step(action)  # observe the results from the action

        agent.add(current_state, action, reward, done, next_state)

        eps_reward += reward

        current_state = next_state

        agent.train()
    return eps_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = PolicyGradient(env)
    logger = logger.Logger()

    for _ in range(agent.max_episodes):
        r = step(agent)
        # if agent.epsilon > agent.epsilon_end:
        #     agent.epsilon *= agent.epsilon_decay
        logger.log(r)

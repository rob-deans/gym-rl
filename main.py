from rl import DeepQAgent
from rl import PolicyGradient
from rl import ActorCritic
from atari import AtariDQN
import logger
import yaml
import numpy as np
import tensorflow as tf
from statistics import Statistics


def step(agent, use_win_condition=False):
    done = False
    agent.reset()

    while not done:

        reward, done = agent.step()
        agent.eps_reward += reward

    agent.total_rewards.append(agent.eps_reward)
    if use_win_condition and len(agent.total_rewards) >= agent.win_condition_over:
        agent.won = np.mean(agent.rewards[-agent.win_condition_over:]) >= agent.win_condition_score


def get_agent(agent_type, config, env):
    if agent_type == 'dqn':
        agent = DeepQAgent(config, env)
    elif agent_type == 'policy':
        agent = PolicyGradient(config, env)
    elif agent_type == 'actor_critic':
        agent = ActorCritic(config, env)
    elif agent_type == 'atari-dqn':
        agent = AtariDQN(config, env)
    else:
        raise NotImplementedError
    return agent


if __name__ == '__main__':
    with open("config.yaml", 'r') as stream:
        config = yaml.load(stream)

    stats = Statistics(config)
    logger = logger.Logger()
    general = config['general']
    if not general['test']:

        active_agent = general['active_agent']
        active_env = general['active_env']
        win_condition = general['use_win_condition']
        agent = get_agent(active_agent, config, active_env)

        for _ in range(agent.max_episodes):
            step(agent, win_condition)
            logger.log(agent.eps_reward)
            if agent.won:
                break
    else:
        envs_to_test = config['test']['envs']
        for env in envs_to_test:

            methods = config['test']['methods']
            for method in methods:

                number_of_tests = config['test']['number']
                for i in range(number_of_tests):

                    tf.reset_default_graph()
                    current_agent = get_agent(method, config, env)

                    for _ in range(current_agent.max_episodes):

                        step(current_agent)
                        logger.log(current_agent.eps_reward)

                    stats.add(current_agent, env)
                    logger.reset()

        stats.save()
        stats.visualise()

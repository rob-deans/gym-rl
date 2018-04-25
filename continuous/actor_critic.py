import tensorflow as tf
import numpy as np
from rl.agent import BaseAgent
import random
from collections import deque


class ACContinuous(BaseAgent):
    def __init__(self, config, env):
        super(ACContinuous, self).__init__(config, env, 'ac_continuous')

        # ==================== #
        #    Hyper parameters  #
        # ==================== #
        self.actor_lr = self.get_attribute('actor_lr')
        self.critic_lr = self.get_attribute('critic_lr')
        self.gamma = self.get_attribute('gamma')

        # ==================== #
        #        Memory        #
        # ==================== #
        self.memory = deque(maxlen=self.get_attribute('max_memory_size'))
        self.batch_size = self.get_attribute('batch_size')

        # ==================== #
        #        Network       #
        # ==================== #
        self.value_size = 1
        self.actor_input_state, self.actor_input_action, self.actor_td_error, self.actor_output, \
            self.critic_input_state, self.critic_td_target, self.critic_output, self.normal_dist = self.create_network()

        self.actor_optimise, self.critic_optimise = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network(self):
        # ============================ #
        #            Actor             #
        # ============================ #
        actor_input_state = tf.placeholder(tf.float32, shape=[None, self.state_size], name='actor_i_state')
        actor_input_action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actor_i_act')
        actor_td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_placeholder')

        init = tf.random_normal_initializer(0., .01)

        net = tf.layers.dense(inputs=actor_input_state, units=36, activation=tf.nn.relu, kernel_initializer=init,
                              name='dense_1')
        sigma = tf.layers.dense(inputs=net, units=1, activation=tf.nn.softplus, kernel_initializer=init)
        mu = tf.layers.dense(inputs=net, units=1, activation=tf.nn.tanh, kernel_initializer=init)

        mu, sigma = mu * 2, sigma + .1
        normal_dist = tf.distributions.Normal(mu, sigma)
        actor_output = tf.clip_by_value(normal_dist.sample(1), -2, 2)

        # ============================ #
        #            Critic            #
        # ============================ #

        critic_input = tf.placeholder(tf.float32, shape=[None, self.state_size])
        critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='critic_td')

        critic_net = tf.layers.dense(inputs=critic_input, units=16, activation=tf.nn.relu, kernel_initializer=init)
        # critic_net = tf.layers.dense(inputs=critic_net, units=15, activation=tf.nn.relu, kernel_initializer=init)
        # critic_net = tf.layers.dense(inputs=critic_net, units=10, activation=tf.nn.relu, kernel_initializer=init)

        critic_output = tf.layers.dense(inputs=critic_net, units=self.value_size, activation=None,
                                        kernel_initializer=init)

        return actor_input_state, actor_input_action, actor_td_error, actor_output, \
            critic_input, critic_td_target, critic_output, normal_dist

    def loss_fn(self):
        # Categorical cross entropy
        loss = self.normal_dist.log_prob(self.actor_input_action) * self.actor_td_error
        loss += 0.01*self.normal_dist.entropy()

        actor_optimise = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_td_target, self.critic_output))
        critic_optimise = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)

        return actor_optimise, critic_optimise

    def step(self, render=False):
        if render:
            self.env.render()

        action = self.get_action(self.current_state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        self.add(self.current_state, action, reward, done, next_state)

        self.current_state = next_state

        self.train()

        return reward, done

    def run(self, states):
        return self.session.run(self.critic_output, feed_dict={self.critic_input_state: states})

    def get_action(self, current_state):
        return self.session.run(self.actor_output, feed_dict={self.actor_input_state: [current_state]})[0][0]

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        samples = self.get()

        td_targets = []
        td_errors = []

        states = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        done = [sample[3] for sample in samples]
        next_states = [sample[4] for sample in samples]

        values = self.run(states)
        value_primes = self.run(next_states)

        for i in range(self.batch_size):
            if done[i]:
                td_targets.append([rewards[i]])
            else:
                td_targets.append(rewards[i] + self.gamma * value_primes[i])

            td_errors.append(td_targets[-1] - values[i])

        self.session.run(self.critic_optimise, feed_dict={self.critic_input_state: states,
                                                          self.critic_td_target: td_targets})

        self.session.run(self.actor_optimise, feed_dict={self.actor_input_state: states,
                                                         self.actor_input_action: actions,
                                                         self.actor_td_error: td_errors})
        pass

    def save(self):
        pass

    def load(self):
        pass

    def add(self, current_state, action, reward, done, next_state):
        self.memory.append([current_state, action, reward, done, next_state])

    def get(self):
        return random.sample(self.memory, self.batch_size)

    def __str__(self):
        return 'ac_continuous'

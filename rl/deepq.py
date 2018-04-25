import tensorflow as tf

import numpy as np
from agent import BaseAgent
import random
from collections import deque


class DeepQAgent(BaseAgent):
    def __init__(self, config, env):
        super(DeepQAgent, self).__init__(config, env, 'dqn')

        # ==================== #
        #    Hyper parameters  #
        # ==================== #
        self.learning_rate = self.get_attribute('learning_rate')
        self.gamma = self.get_attribute('gamma')

        # ==================== #
        #        Memory        #
        # ==================== #
        self.memory = deque(maxlen=self.get_attribute('max_memory_size'))
        self.batch_size = self.get_attribute('batch_size')

        # ==================== #
        #        Epsilon       #
        # ==================== #
        self.epsilon = self.get_attribute('epsilon_start')
        self.epsilon_end = self.get_attribute('epsilon_end')
        self.epsilon_decay = self.get_attribute('epsilon_decay')

        # ==================== #
        #        Network       #
        # ==================== #
        self.states, self.actions, self.rewards, self.output = self.create_network()
        self.optimiser = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network(self):
        states = tf.placeholder(np.float32, shape=[None, self.state_size], name='input')
        actions = tf.placeholder(np.float32, shape=[None, self.num_actions], name='q_values_new')
        rewards = tf.placeholder(np.float32, shape=[None], name='rewards')

        init = tf.truncated_normal_initializer()

        net = tf.layers.dense(inputs=states, units=32, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        net = tf.layers.dense(inputs=net, units=32, activation=tf.nn.relu, kernel_initializer=init, name='dense_2')

        net = tf.layers.dense(inputs=net, units=self.num_actions,
                              kernel_initializer=init, activation=None, name='output')

        return states, actions, rewards, net

    def loss_fn(self):
        q_reward = tf.reduce_sum(tf.multiply(self.output, self.actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))

        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def step(self, render=False):
        if render:
            self.env.render()

        action = self.get_action(self.current_state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        self.add(self.current_state, action, reward, done, next_state)

        self.current_state = next_state

        self.train()

        if done:
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

        return reward, done

    def train(self):

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards = self.get()

        feed_dict = {self.states: states, self.actions: actions, self.rewards: rewards}
        self.session.run(self.optimiser, feed_dict)

    def run(self, state):
        feed_dict = {self.states: state}
        return self.session.run(self.output, feed_dict)

    def get_action(self, current_state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.run([current_state])[0]
            action = np.argmax(q_values)

        return action

    def add(self, current_state, action, reward, done, next_state):
        action = self.one_hot_encode(action)
        self.memory.append([current_state, action, reward, done, next_state])

    def get(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]
        next_states = self.run(next_states)

        y_batch = []

        for i in range(self.batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(next_states[i]))

        return states, actions, y_batch

    def load(self):
        pass

    def save(self):
        pass

    def __str__(self):
        return 'deepq'

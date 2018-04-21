import tensorflow as tf
import numpy as np
from rl.agent import BaseAgent
import random
from collections import deque


class AtariDQN(BaseAgent):
    def __init__(self, config, env):
        super(AtariDQN, self).__init__(config, env, 'dqn')

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
        self.state = deque(maxlen=4)
        [self.state.append(np.zeros(6400)) for _ in range(4)]

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
        self.state1, self.state2, self.state3, self.state4 = self.states
        self.optimiser = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network(self):
        states1 = tf.placeholder(np.float32, shape=[None, self.state_size], name='input1')
        states2 = tf.placeholder(np.float32, shape=[None, self.state_size], name='input2')
        states3 = tf.placeholder(np.float32, shape=[None, self.state_size], name='input3')
        states4 = tf.placeholder(np.float32, shape=[None, self.state_size], name='input4')

        actions = tf.placeholder(np.float32, shape=[None, self.num_actions], name='q_values_new')
        rewards = tf.placeholder(np.float32, shape=[None], name='rewards')

        init = tf.truncated_normal_initializer()

        net = tf.reshape(states1, [-1, 80, 80, 1])
        net1 = tf.reshape(states2, [-1, 80, 80, 1])
        net2 = tf.reshape(states3, [-1, 80, 80, 1])
        net3 = tf.reshape(states4, [-1, 80, 80, 1])

        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        net = tf.contrib.layers.flatten(net)

        net1 = tf.layers.conv2d(inputs=net1, filters=16, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        net1 = tf.layers.conv2d(inputs=net1, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        net1 = tf.contrib.layers.flatten(net1)

        net2 = tf.layers.conv2d(inputs=net2, filters=16, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        net2 = tf.layers.conv2d(inputs=net2, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        net2 = tf.contrib.layers.flatten(net2)

        net3 = tf.layers.conv2d(inputs=net3, filters=16, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        net3 = tf.layers.conv2d(inputs=net3, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        net3 = tf.contrib.layers.flatten(net3)

        net = tf.concat([net, net1, net2, net3], 1)

        net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu, kernel_initializer=init, name='dense')

        net = tf.layers.dense(inputs=net, units=self.num_actions,
                              kernel_initializer=init, activation=None, name='output')

        # return states1, actions, rewards, net
        return (states1, states2, states3, states4), actions, rewards, net

    def loss_fn(self):
        q_reward = tf.reduce_sum(tf.multiply(self.output, self.actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))

        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def step(self, render=False):
        if render:
            self.env.render()

        self.current_state = self.preprocess(self.current_state)
        self.state.append(self.current_state)

        action = self.get_action(self.state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        ns = self.preprocess(next_state)
        new_queue = deque(list(self.state)[:], maxlen=4)
        new_queue.append(ns)

        self.add(self.state, action, reward, done, new_queue)

        self.current_state = next_state

        self.train()

        if done:
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

        return reward, done

    def preprocess(self, state):
        # self.current_state = self.current_state.flatten()[:, 0]
        state = state[35:195]  # crop
        state = state[::2, ::2, 0]  # downsample by factor of 2
        # state[state == 144] = 0  # erase background (background type 1)
        # state[state == 109] = 0  # erase background (background type 2)
        # state[state != 0] = 1  # everything else (paddles, ball) just set to 1

        return state.astype(np.float).ravel()

    def train(self):

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards = self.get()
        states = np.array(states)

        feed_dict = {self.state1: states[:, 0],
                     self.state2: states[:, 1],
                     self.state3: states[:, 2],
                     self.state4: states[:, 3],
                     self.actions: actions, self.rewards: rewards}
        self.session.run(self.optimiser, feed_dict)

    def run(self, state):
        feed_dict = {self.state1: [state[0]],
                     self.state2: [state[1]],
                     self.state3: [state[2]],
                     self.state4: [state[3]]}
        return self.session.run(self.output, feed_dict)[0]

    def get_action(self, current_state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.run(current_state)
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

        y_batch = []

        for i in range(self.batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(self.run(next_states[i])))

        return states, actions, y_batch

    def load(self):
        pass

    def save(self):
        pass

    def __str__(self):
        return 'atari-deepq'

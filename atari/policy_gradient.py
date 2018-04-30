import tensorflow as tf
import numpy as np
from agent import BaseAgent
from collections import deque
import os


class AtariPolicy(BaseAgent):
    def __init__(self, config, env):
        super(AtariPolicy, self).__init__(config, env, 'atari-policy')

        # ==================== #
        #    Hyper parameters  #
        # ==================== #
        self.learning_rate = self.get_attribute('learning_rate')
        self.gamma = self.get_attribute('gamma')

        # ==================== #
        #        Memory        #
        # ==================== #
        self.batch_size = self.get_attribute('batch_size')
        self.episode_rewards = []
        self.b_obs, self.b_acts, self.b_rews = [], [], []
        self.state = deque(maxlen=4)
        [self.state.append(np.zeros(6400)) for _ in range(4)]

        # ==================== #
        #        Network       #
        # ==================== #
        self.states, self.actions, self.advantages, self.output, self.logits = self.create_network()
        # self.state1, self.state2, self.state3, self.state4 = self.states
        self.optimiser = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.prev_x = None

    def create_network(self):
        states = tf.placeholder(np.float32, shape=[None, self.state_size], name='p_input')
        actions = tf.placeholder(np.float32, shape=[None, self.num_actions])
        advantages = tf.placeholder(np.float32, shape=[None])

        init = tf.random_normal_initializer

        net = tf.reshape(states, [-1, 80, 80, 1])

        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu, kernel_initializer=init, name='dense_1')
        # net = tf.layers.dense(inputs=net, units=24, activation=tf.nn.relu, kernel_initializer=init, name='dense_2')
        logits = tf.layers.dense(inputs=net, units=self.num_actions, activation=None,
                                 kernel_initializer=init, name='output')
        logits = tf.clip_by_value(logits, 1e-24, 1.)
        softmax_logits = tf.nn.softmax(logits)

        return states, actions, advantages, softmax_logits, logits

    def loss_fn(self):
        loss = -tf.log(tf.reduce_sum(tf.multiply(self.actions, self.output), reduction_indices=1)) * self.advantages
        # temp = [np.argmax(action) for action in self.actions]
        # loss -= 0.01 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=temp, logits=self.logits)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def step(self, render=False):
        if render:
            self.env.render()

        cur_x = self.preprocess(self.current_state)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(80*80)

        self.prev_x = cur_x

        action = self.get_action(x)

        self.current_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        self.add(x, action, reward, None, None)

        if done:
            advantages = self.discount_rewards(self.episode_rewards)
            self.b_rews.extend(advantages)

            self.episode_rewards = []

            if self.episode % self.batch_size == 0 and self.episode > 0:
                self.train()
            if self.episode % 100 == 0:
                self.save()

        return reward, done

    def preprocess(self, state):
        state = state[35:195]  # crop
        state = state[::2, ::2, 0]  # downsample by factor of 2
        state[state == 144] = 0  # erase background (background type 1)
        state[state == 109] = 0  # erase background (background type 2)
        state[state != 0] = 1  # everything else (paddles, ball) just set to 1
        return state.astype(np.float).ravel()

    def run(self, state):
        feed_dict = {self.states: [state]}
        return self.session.run(self.output, feed_dict=feed_dict)[0]

    def get_action(self, current_state):
        output = self.run(current_state)
        return np.random.choice(self.num_actions, 1, p=output)[0]

    def train(self):
        states, actions, ads = self.get()
        states = np.array(states)

        feed_dict = {self.states: states,
                     self.actions: actions,
                     self.advantages: ads}
        self.session.run(self.optimiser, feed_dict=feed_dict)
        self.clear_memory()

    def save(self):
        current_file = os.path.dirname(__file__)
        self.saver.save(self.session, current_file + 'policy_atari_pong.ckpt')
        pass

    def load(self):
        pass

    def add(self, current_state, action, reward, done, next_state):
        self.b_obs.append(current_state)
        action = self.one_hot_encode(action)
        self.b_acts.append(action)
        self.episode_rewards.append(reward)

    def get(self):
        self.b_rews = (self.b_rews - np.mean(self.b_rews)) // (np.std(self.b_rews) + 1e-10)
        return self.b_obs, self.b_acts, self.b_rews

    def clear_memory(self):
        self.b_obs, self.b_acts, self.b_rews = [], [], []
        self.prev_x = None

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        return discounted_r

    def __str__(self):
        return 'atari-policy'

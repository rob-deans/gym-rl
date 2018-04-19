import tensorflow as tf
import numpy as np
from agent import BaseAgent


class PolicyGradient(BaseAgent):
    def __init__(self, env):
        super(PolicyGradient, self).__init__(env)

        # ==================== #
        #       Env stuff      #
        # ==================== #
        self.max_episodes = 400

        # ==================== #
        #    Hyper parameters  #
        # ==================== #
        self.learning_rate = 1e-1

        # ==================== #
        #        Memory        #
        # ==================== #
        self.batch_size = 10
        self.episode_rewards = []
        self.b_obs, self.b_acts, self.b_rews = [], [], []

        # ==================== #
        #        Network       #
        # ==================== #
        self.states, self.actions, self.advantages, self.output, self.logits = self.create_network()
        self.optimiser = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.learning_rate = 1e-1

    def create_network(self):
        states = tf.placeholder(np.float32, shape=[None, self.state_size], name='input')
        actions = tf.placeholder(np.int32)
        advantages = tf.placeholder(np.float32)

        hidden = tf.contrib.layers.fully_connected(
            inputs=states,
            num_outputs=36,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer
        )

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden,
            num_outputs=self.num_actions,
            activation_fn=None
        )
        _sample = tf.reshape(tf.multinomial(logits, 1), [])

        return states, actions, advantages, _sample, logits

    def loss_fn(self):
        log_prob = tf.log(tf.nn.softmax(self.logits))
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)
        loss = -tf.reduce_sum(tf.multiply(act_prob, self.advantages))

        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def step(self, render=False):
        if render:
            self.env.render()

        action = self.get_action(self.current_state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        self.add(self.current_state, action, reward, None, None)

        self.current_state = next_state

        if done:
            advantages = self.process_rewards(self.episode_rewards)
            self.b_rews.extend(advantages)

            self.episode_rewards = []

            if self.episode % self.batch_size == 0 and self.episode > 0:
                self.train()

        return reward, done

    def run(self, state):
        return self.session.run(self.output, {self.states: [state]})

    def get_action(self, current_state):
        return self.run(current_state)

    def train(self):
        print('== TRAINING ==')
        states, actions, ads = self.get()
        self.session.run(self.optimiser, feed_dict={self.states: states, self.actions: actions, self.advantages: ads})
        self.clear_memory()

    def save(self):
        pass

    def load(self):
        pass

    def add(self, current_state, action, reward, done, next_state):
        self.b_obs.append(self.current_state)
        # action = self.one_hot_encode(action)
        self.b_acts.append(action)
        self.episode_rewards.append(reward)

    def get(self):
        self.b_rews = (self.b_rews - np.mean(self.b_rews)) // (np.std(self.b_rews) + 1e-10)
        return self.b_obs, self.b_acts, self.b_rews

    def clear_memory(self):
        self.b_obs, self.b_acts, self.b_rews = [], [], []

    @staticmethod
    def process_rewards(rewards):
        """Rewards -> Advantages for one episode. """

        # total reward: length of episode
        return [len(rewards)] * len(rewards)

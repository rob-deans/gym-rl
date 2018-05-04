import tensorflow as tf
import numpy as np
from agent import BaseAgent
import random
from collections import deque


class ActorCritic(BaseAgent):
    def __init__(self, config, env):
        super(ActorCritic, self).__init__(config, env, 'actor_critic')

        self.demo = config['general']['load']

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
        self.input_state, self.actor_input_action, self.actor_td_error, self.actor_output, self.critic_input_state, \
            self.critic_td_target, self.critic_output = self.create_network()

        self.actor_optimise, self.critic_optimise = self.loss_fn()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.demo:
            self.saver.restore(self.session, '/home/rob/Documents/uni/fyp/gym-rl/ac_model.ckpt')

    def create_network(self):
        # ============================ #
        #            Actor             #
        # ============================ #
        input_state = tf.placeholder(tf.float32, shape=[None, self.state_size], name='actor_i_state')
        actor_input_action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actor_i_act')
        actor_td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_placeholder')

        # init = tf.truncated_normal_initializer(0, 0.01)
        init = tf.random_normal_initializer

        net = tf.layers.dense(inputs=input_state, units=36, activation=tf.nn.relu, kernel_initializer=init,
                              name='dense_1')
        # net = tf.layers.dense(inputs=net, units=32, activation=tf.nn.relu, kernel_initializer=init,
        #                       name='dense_2')

        actor_output = tf.layers.dense(inputs=net, units=self.num_actions, kernel_initializer=init,
                                       activation=tf.nn.softmax, name='output')

        actor_output = tf.clip_by_value(actor_output, 1e-24, 1.)

        # ============================ #
        #            Critic            #
        # ============================ #

        critic_input = tf.placeholder(tf.float32, shape=[None, self.state_size], name='critic_input')
        critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='critic_td')

        critic_net = tf.layers.dense(inputs=critic_input, units=32, activation=tf.nn.relu, kernel_initializer=init)
        # critic_net = tf.layers.dense(inputs=critic_net, units=32, activation=tf.nn.relu, kernel_initializer=init)

        critic_output = tf.layers.dense(inputs=critic_net, units=self.value_size, activation=None,
                                        kernel_initializer=init)

        return input_state, actor_input_action, actor_td_error, actor_output, \
            critic_input, critic_td_target, critic_output

    def loss_fn(self):
        # Categorical cross entropy
        loss = tf.log(tf.reduce_sum(tf.multiply(self.actor_input_action, self.actor_output))) * self.actor_td_error
        actor_optimise = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_td_target, self.critic_output))
        critic_optimise = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)

        return actor_optimise, critic_optimise

    def step(self, render=False):
        if render:
            self.env.render()

        action = self.get_action(self.current_state)

        next_state, reward, done, _ = self.env.step(action)  # observe the results from the action

        r = reward
        if done and self.eps_reward < 200 and self.env_name == 'cartpole':
            reward = -100

        self.add(self.current_state, action, reward, done, next_state)

        self.current_state = next_state

        if not self.demo:
            self.train()

        return r, done
        # return reward if reward != -100 else reward + 100, done

    def run(self, states):
        return self.session.run(self.critic_output, feed_dict={self.critic_input_state: states})

    def get_action(self, current_state):
        actions = self.session.run(self.actor_output, feed_dict={self.input_state: [current_state]})[0]
        # return np.argmax(np.random.multinomial(1, actions))
        if 'nan' == str(actions[0]) or 'nan' == str(actions[1]):
            print(self.num_actions)
        return np.random.choice(self.num_actions, 1, p=actions)[0]

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

        self.session.run(self.actor_optimise, feed_dict={self.input_state: states,
                                                         self.actor_input_action: actions,
                                                         self.actor_td_error: td_errors})

    def save(self):
        self.saver.save(self.session,  '/home/rob/Documents/uni/fyp/gym-rl/ac_model.ckpt')

    def load(self):
        pass

    def add(self, current_state, action, reward, done, next_state):
        action = self.one_hot_encode(action)
        self.memory.append([current_state, action, reward, done, next_state])

    def get(self):
        return random.sample(self.memory, self.batch_size)

    def __str__(self):
        return 'actor_critic'

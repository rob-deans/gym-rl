import tensorflow as tf
import numpy as np
from rl.agent import BaseAgent
import random
from collections import deque
import scipy


class AtariDQN(BaseAgent):
    def __init__(self, config, env):
        super(AtariDQN, self).__init__(config, env, 'atari-dqn')

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
        self.saver = tf.train.Saver()

        self.motion_tracer = None

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
            self.motion_tracer = None
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            if self.episode % 100 == 0:
                self.save()

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
        self.saver.save(self.session, '/home/rob/Documents/uni/fyp/gym-rl/dqn_atari_pong.ckpt')

    def __str__(self):
        return 'atari-deepq'


class MotionTracer:
    """
    Used for processing raw image-frames from the game-environment.
    The image-frames are converted to gray-scale, resized, and then
    the background is removed using filtering of the image-frames
    so as to detect motions.
    This is needed because a single image-frame of the game environment
    is insufficient to determine the direction of moving objects.

    The original DeepMind implementation used the last 4 image-frames
    of the game-environment to allow the Neural Network to learn how
    to detect motion. This implementation could make it a little easier
    for the Neural Network to learn how to detect motion, but it has
    only been tested on Breakout and Space Invaders, and may not work
    for games with more complicated graphics such as Doom. This remains
    to be tested.
    """

    def __init__(self, image, decay=0.75):
        """

        :param image:
            First image from the game-environment,
            used for resetting the motion detector.
        :param decay:
            Parameter for how long the tail should be on the motion-trace.
            This is a float between 0.0 and 1.0 where higher values means
            the trace / tail is longer.
        """

        # Pre-process the image and save it for later use.
        # The input image may be 8-bit integers but internally
        # we need to use floating-point to avoid image-noise
        # caused by recurrent rounding-errors.
        img = _pre_process_image(image=image)
        self.last_input = img.astype(np.float)

        # Set the last output to zero.
        self.last_output = np.zeros_like(img)

        self.decay = decay

    def process(self, image):
        """Process a raw image-frame from the game-environment."""

        # Pre-process the image so it is gray-scale and resized.
        img = _pre_process_image(image=image)

        # Subtract the previous input. This only leaves the
        # pixels that have changed in the two image-frames.
        img_dif = img - self.last_input

        # Copy the contents of the input-image to the last input.
        self.last_input[:] = img[:]

        # If the pixel-difference is greater than a threshold then
        # set the output pixel-value to the highest value (white),
        # otherwise set the output pixel-value to the lowest value (black).
        # So that we merely detect motion, and don't care about details.
        img_motion = np.where(np.abs(img_dif) > 20, 255.0, 0.0)

        # Add some of the previous output. This recurrent formula
        # is what gives the trace / tail.
        output = img_motion + self.decay * self.last_output

        # Ensure the pixel-values are within the allowed bounds.
        output = np.clip(output, 0.0, 255.0)

        # Set the last output.
        self.last_output = output

        return output

    def get_state(self):
        """
        Get a state that can be used as input to the Neural Network.
        It is basically just the last input and the last output of the
        motion-tracer. This means it is the last image-frame of the
        game-environment, as well as the motion-trace. This shows
        the current location of all the objects in the game-environment
        as well as trajectories / traces of where they have been.
        """

        # Stack the last input and output images.
        state = np.dstack([self.last_input, self.last_output])

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state


def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray


def _pre_process_image(image):
    """Pre-process a raw image from the game-environment."""

    # Convert image to gray-scale.
    img = _rgb_to_grayscale(image)

    # Resize to the desired size using SciPy for convenience.
    img = scipy.misc.imresize(img, size=state_img_size, interp='bicubic')

    return img

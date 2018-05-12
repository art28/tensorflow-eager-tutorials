from __future__ import print_function

import random
from glob import glob

import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from replay_memory import ReplayMemory

# eager execution
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

# Hyper parameter
INITIAL_EPSILON = 1.0  # initial exploration rate
FINAL_EPSILON = 0.05  # final exploration rate
LEARNING_RATE = 0.001  # learning rate
OBSERVATION_STEPS = 50000  # step for observing(not trainig)
EXPLORATION_STEPS = 500000  # step for exploration(epsilon > FINAL_EPSILON)
BATCH_SIZE = 32  # batch size
GAMMA = 0.97  # discount rate


class DQNAgent(tf.keras.Model):
    def __init__(self,
                 state_shape=(-1,80,80,1),
                 action_dim=4,
                 checkpoint_directory="models_checkpoints/rl/",
                 batch_size=BATCH_SIZE,
                 initial_epsilon= INITIAL_EPSILON,
                 final_epsilon=FINAL_EPSILON,
                 exploration_steps=EXPLORATION_STEPS,
                 observation_steps=OBSERVATION_STEPS,
                 device_name='cpu:0'):

        super(DQNAgent, self).__init__()
        # state's shape , in Atari we will use (-1, 105, 80, 1)
        self.state_shape = state_shape
        # number of actions, in Atari 4
        self.action_dim = action_dim
        # saving checkpoint directory
        self.checkpoint_directory = checkpoint_directory

        self.observation_steps=observation_steps
        self.exploration_steps = exploration_steps
        self.initial_epsilon=initial_epsilon
        self.final_epsilon=final_epsilon

        # init q layers
        self.conv1 = tf.layers.Conv2D(32, 8, 8, padding='same', activation=tf.nn.relu)
        self.batch1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(64, 4, 4, padding='same', activation=tf.nn.relu)
        self.batch2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(64, 3, 3, padding='same', activation=tf.nn.relu)
        self.flatten = tf.layers.Flatten()

        self.dense1 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(action_dim, activation=None)

        self.base_layers = [self.conv1, self.batch1, self.conv2, self.batch2, self.conv3, self.flatten, self.dense1,
                            self.dense2]

        # target q layers
        self.conv1_t = tf.layers.Conv2D(32, 8, 8, padding='same', activation=tf.nn.relu)
        self.batch1_t = tf.layers.BatchNormalization()
        self.conv2_t = tf.layers.Conv2D(64, 4, 4, padding='same', activation=tf.nn.relu)
        self.batch2_t = tf.layers.BatchNormalization()
        self.conv3_t = tf.layers.Conv2D(64, 3, 3, padding='same', activation=tf.nn.relu)
        self.flatten_t = tf.layers.Flatten()

        self.dense1_t = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dense2_t = tf.layers.Dense(action_dim, activation=None)

        self.target_layers = [self.conv1_t, self.batch1_t, self.conv2_t, self.batch2_t, self.conv3_t, self.flatten_t,
                              self.dense1_t, self.dense2_t]

        # learning optimizer
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        # epsilon-greedy
        self.epsilon = initial_epsilon
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps

        # replay_memory
        self.replay_memory = ReplayMemory(100000)
        self.batch_size = batch_size

        # for logging
        self.step_count = 0
        self.sum_loss = 0;

        # device configuration
        self.device_name = device_name

    def predict(self, state_batch, training):

        # you can use prediction with numpy array state input
        if isinstance(state_batch, (np.ndarray, np.generic)):
            state_batch = np.reshape(state_batch, self.state_shape)
            state_batch = tf.convert_to_tensor(state_batch)

        x = self.conv1(state_batch)
        x = self.batch1(x, training=training)
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

    def predict_target(self, state_batch, training):

        # you can use prediction with numpy array state input
        if isinstance(state_batch, (np.ndarray, np.generic)):
            state_batch = np.reshape(state_batch, self.state_shape)
            state_batch = tf.convert_to_tensor(state_batch)

        x = self.conv1_t(state_batch)
        x = self.batch1_t(x, training=training)
        x = self.conv2_t(x)
        x = self.batch2_t(x, training=training)
        x = self.conv3_t(x)
        x = self.flatten_t(x)
        x = self.dense1_t(x)
        x = self.dense2_t(x)

        return x

    def copy_base_to_target(self):
        """copy base's weights to target"""
        for idx_layer in range(len(self.base_layers)):
            base = self.base_layers[idx_layer]
            target = self.target_layers[idx_layer]
            for idx_weight in range(len(base.weights)):
                tf.assign(target.weights[idx_weight], base.weights[idx_weight])
            if hasattr(base, "bias"):
                tf.assign(target.bias, base.bias)

    @staticmethod
    def huber_loss(labels, predictions):
        error = labels - predictions
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = tf.convert_to_tensor((abs(error) > 1.0).numpy().astype("float32"))

        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

    def loss(self, state_batch, target, training):
        predictions = self.predict(state_batch, training)
        # loss_value = tf.losses.mean_squared_error(labels=target, predictions=predictions)
        loss_value = self.huber_loss(labels=target, predictions=predictions)
        self.sum_loss += tf.reduce_sum(loss_value).numpy()
        return loss_value

    def grad(self, state_batch, target, training):
        with tfe.GradientTape() as tape:
            loss_value = self.loss(state_batch, target, training)
        return tape.gradient(loss_value, self.variables)

    def get_action(self, state, training=False):
        if training:
            if self.epsilon >= random.random():
                action = tf.convert_to_tensor(random.randrange(self.action_dim))
            else:
                action = tf.argmax(self.predict(state, training=training), 1)

            if self.epsilon > self.final_epsilon and self.step_count > self.observation_steps:
                self.epsilon -= self.epsilon_step

            return action

        else:
            return tf.argmax(self.predict(state, training=training), 1)

    def step(self, state, action, reward, next_state, terminal):
        if self.step_count <= self.observation_steps:
            self.observe(state, action, reward, next_state, terminal)
        else:
            self.fit(state, action, reward, next_state, terminal)

        if self.step_count % 1000 == 0:
            print("STEP %s : EPSILON [%6f]...." % (self.step_count, self.epsilon))
            print("=============================================")
        self.step_count += 1

    def observe(self, state, action, reward, next_state, terminal):
        self.replay_memory.add(state, action, reward, next_state, terminal)

    def fit(self, state, action, reward, next_state, terminal, num_epochs=1):

        self.replay_memory.add(state, action, reward, next_state, terminal)

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_memory.get_batch(
            self.batch_size)

        current_q = self.predict(state_batch, training=False).numpy()
        now_q = np.zeros((self.batch_size,self.action_dim))

        target_q_batch = self.predict_target(next_state_batch, training=False)

        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_batch, axis=1)

        for i in range(self.batch_size):
            now_q[i, action_batch[i]] = y_batch[i]

        with tf.device(self.device_name):
            for i in range(num_epochs):
                grads = self.grad(state_batch, now_q, True)
                self.optimizer.apply_gradients(zip(grads, self.variables))

        if self.step_count % 1000 == 0 :
            print("loss: %6f" % (self.sum_loss/1000))
            self.sum_loss = 0
            # print(current_q[0])
            # print(now_q[0])
            print(self.predict(state_batch[0], training=False).numpy())

        return

    def save(self, global_step=0):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=global_step)

    def load_last_checkpoint(self):
        # Run the model once to initialize variables
        initialshape = list(self.state_shape)
        initialshape[0] = 1
        initialshape = tuple(initialshape)
        dummy_input = tf.constant(tf.zeros(initialshape))
        dummy_pred = self.predict(dummy_input, training=False)
        # Restore the variables of the model
        saver = tfe.Saver(self.variables)
        saver.restore(tf.train.latest_checkpoint
                      (self.checkpoint_directory))

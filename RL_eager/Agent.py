from __future__ import print_function

import random

import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ReplayMemory import ReplayMemory

tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

# Hyper parameter
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
LEARNING_RATE = 0.001
EXPLORATION_STEPS = 1000
BATCH_SIZE = 32
GAMMA = 0.95


class DQNAgent(tf.keras.Model):
    def __init__(self, state_shape, action_dim, checkpoint_directory, batch_size=32):
        super(DQNAgent, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim

        self.checkpoint_directory = checkpoint_directory

        # init q layers
        self.conv1 = tf.layers.Conv2D(32, 8, 8, padding='same', activation=tf.nn.relu)
        self.batch1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(64, 4, 4, padding='same', activation=tf.nn.relu)
        self.batch2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(64, 3, 3, padding='same', activation=tf.nn.relu)
        self.flatten = tf.layers.Flatten()

        self.dense1 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(action_dim, activation=None)

        # target q layers
        self.conv1_t = tf.layers.Conv2D(32, 8, 8, padding='same', activation=tf.nn.relu)
        self.batch1_t = tf.layers.BatchNormalization()
        self.conv2_t = tf.layers.Conv2D(64, 4, 4, padding='same', activation=tf.nn.relu)
        self.batch2_t = tf.layers.BatchNormalization()
        self.conv3_t = tf.layers.Conv2D(64, 3, 3, padding='same', activation=tf.nn.relu)
        self.flatten_t = tf.layers.Flatten()

        self.dense1_t = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dense2_t = tf.layers.Dense(action_dim, activation=None)
        # learning optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        # epsilon-greedy
        self.epsilon = tfe.Variable(INITIAL_EPSILON)
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        # replay_memory
        self.replay_memory = ReplayMemory(10000)
        self.batch_size = batch_size

    def predict(self, state_batch, training):

        if isinstance(state_batch, (np.ndarray, np.generic)):
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

    def loss(self, state_batch, target, training):
        preds = self.predict(state_batch, training)
        loss_value = tf.losses.mean_squared_error(labels=target, predictions=preds)
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
                action = tf.argmax(self.predict(state.reshape(-1, 105, 80, 1), training=training), 1)

            if self.epsilon > FINAL_EPSILON:
                self.epsilon.assign_sub(self.epsilon_step)

            return action

        else:
            return tf.argmax(self.predict(state.reshape(-1, 105, 80, 1), training=training), 1)

    def fit(self, state, action, reward, next_state, terminal, num_epochs=1):

        self.replay_memory.add(state, action, reward, next_state, terminal)

        if len(self.replay_memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_memory.get_batch(
            self.batch_size)

        current_q = self.predict(state_batch, training=False).numpy()
        now_q = current_q.copy() * 0.75

        target_q_batch = self.predict(next_state_batch, training=False)

        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_batch, axis=1)

        for i in range(self.batch_size):
            now_q[i, action_batch[i]] = y_batch[i]

        # if(terminal_batch[0]):
        #             print("r" , reward_batch[0])
        #             print("t" , terminal_batch[0])
        #             print("q" , np.max(target_q_batch, axis = 1)[0])
        #             print("s" , current_q[0])
        #             print("y" , now_q[0])

        for i in range(num_epochs):
            grads = self.grad(state_batch, now_q, True)
            self.optimizer.apply_gradients(zip(grads, self.variables))

    def save(self, global_step=0):
        #         print("saving...%i........." % global_step , end='')
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=global_step)

    #         print("saved")

    def load(self):
        # Run the model once to initialize variables
        dummy_input = tf.constant(tf.zeros(self.state_shape))
        dummy_pred = self.predict(dummy_input, training=False)
        # Restore the variables of the model
        saver = tfe.Saver(self.variables)
        saver.restore(tf.train.latest_checkpoint
                      (self.checkpoint_directory))
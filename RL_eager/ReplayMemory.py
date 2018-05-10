from collections import deque
import random
import numpy as np


class ReplayMemory:
    def __init__(self, max_length):
        self.memory = deque(maxlen=max_length)

    def add(self, state, action, reward, next_state, terminal):
        self.memory.append([state, action, reward, next_state, terminal])

    def get_batch(self, batch_size):
        sampling = np.array(random.sample(self.memory, batch_size))
        state_batch = np.stack(sampling[:, 0])
        next_state_batch = np.stack(sampling[:, 3])
        return state_batch, sampling[:, 1], sampling[:, 2], next_state_batch, sampling[:, 4]

    def __len__(self):
        return len(self.memory)
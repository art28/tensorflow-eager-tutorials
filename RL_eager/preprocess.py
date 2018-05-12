import numpy as np
import cv2


def preprocess(observation):
    # RGB to Gray-Scale, to size 80*80
    observation = cv2.cvtColor(cv2.resize(observation, (80, 105)), cv2.COLOR_BGR2GRAY)[25:,:]
    return np.reshape(observation, (80, 80, 1)).astype("float32")
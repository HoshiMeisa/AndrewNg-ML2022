import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
plt.style.use('./deeplearning.mplstyle')


logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

x_train = np.array([[1.0], [2.0]], dtype=np.float32)
y_train = np.array([[300.0], [500.0]], dtype=np.float32)

linear_layer = tf.keras.layers.Dence(unit=1, activation='linear', )
linear_layer.get_weights()

a1 = linear_layer(x_train.reshape(1, 1))
print(a1)

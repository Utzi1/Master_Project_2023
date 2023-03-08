"""
File: sampling_layer.py
Author: ULR
Email: b21utzri@student.his.se
Github: Utzi1
Description: A sampling layer class for VAE's inheriting from keras Layer
"""
import tensorflow as tf

class Sampling_Layer(tf.keras.layers.Layer):
    """
    A sampling layer using the mean and log(var) to sample from an input
    """
    def call(self, inputs):
        mean, logvar = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), mean=0., stddev=1.)
        return mean + tf.exp(logvar * .5) * epsilon

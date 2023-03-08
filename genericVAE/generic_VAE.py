"""
File: generic_VAE.py
Author: ULR
Email: b21utzri@student.his.se
Github: Utzi1
Description: A few classes that implement variable VAE build
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras.backend as K
import sampling_layer



class Builder():
    """
    will return an encoder and a decoder as keras model

    input_shape: a tuple of integers like (32,) or (3, 4)
    encoder_shape: the shape of the encoder which will be handed as a list
    decoder_shape: the shape of the decoder which will be handed as a list
    latent dims: shape of the latent space sampled from the z-vectors
    dropout_rate: 0 by default, otherwise the drouput

    """
    def __init__(self,
                 input_shape,
                 encoder_shape,
                 decoder_shape,
                 latent_dims=2,
                 dropout_rate=0
                 ):
        self.input_shape = input_shape
        self.encoder_shape = encoder_shape
        self.decoder_shape = decoder_shape
        self.latent_dims = latent_dims
        self.dropout_rate = dropout_rate
        # and then call the builder
        self._build()

    def _build(self):
        """
        Builds the encoder an the decoder and keeps them as attribute

        ######################################################################

        """
        encoder_input = keras.layers.Input(self.input_shape)
        # good'ol stack
        layer_stack = encoder_input
        # next we can build encoder
        if self.dropout_rate != 0:
            layer_stack = keras.layers.Dropout(rate=self.dropout_rate)(
                    encoder_input)

        for item in self.encoder_shape:
            layer_stack = keras.layers.Dense(units=item,
                                             activation="relu")(layer_stack)
        # introduce the sampling layers
        # where z is a vector encoding a sampled digit
        z_mean = keras.layers.Dense(units=self.latent_dims, name="mean")(
                layer_stack
                )
        z_logvar = keras.layers.Dense(units=self.latent_dims, name="log_var")(
                layer_stack
                )
        # make a vector encoding this stuff:
        z = sampling_layer.Sampling_Layer()([z_mean, z_logvar])
        # and build the model:
        self.encoder_model = keras.Model(encoder_input, [
            z_mean, z_logvar, z
            ],
                                         name="encoder")

        # to build the decoder:
        # put in the latent stuff
        decoder_input = keras.Input((self.latent_dims, ))
        layer_stack = decoder_input
        for item in self.decoder_shape:
            layer_stack = keras.layers.Dense(units=item,
                                             activation="relu")(layer_stack)

        # last layer needs to be identical to the input
        decoder_output = keras.layers.Dense(self.input_shape[0],
                                            activation="relu")(layer_stack)

        self.decoder_model = keras.Model(decoder_input, decoder_output)



class VAE(keras.Model):
    """
    Will return the complete model with a custom trining-step
    """
    def __init__(self, decoder, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.metrics.Mean(
                name="reconstruction_loss_tracker"
                )
        self.kl_loss_tracker = tf.metrics.Mean(name="kl_loss_tracker")

    @property
    def metrics(self):
        return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker
                ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = K.mean(K.square(data - reconstruction),
                                         axis=1)
            kl_loss = -.5 * (1 + z_logvar - K.square(z_mean) - K.exp(
                z_logvar)
                             )
            kl_loss = K.sum(kl_loss)
            total_loss = kl_loss + reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(
                    ),
                "kl_loss": self.kl_loss_tracker.result(),
                }

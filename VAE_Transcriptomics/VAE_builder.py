#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model, Sequential
from tensorflow.keras import Input
from keras.layers import Dense, LeakyReLU, Dropout, Lambda
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from keras import backend as K
import os
import pickle
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import regularizers

"""
Created on Wed Aug 10 10:09:23 2022

@author: dirk
"""

disable_eager_execution()

class VAE_class():
    def __init__(self,
                  input_shape,
                  layer_shapes_encoder,
                  layer_shapes_decoder,
                  latent_space_shape,
                  dropout_rate,
                  model_name,
                  regularizer = 1e-8):
         
         self.input_shape = input_shape
         self.layer_shapes_encoder = layer_shapes_encoder
         self.layer_shapes_decoder = layer_shapes_decoder
         self.latent_space_shape = latent_space_shape
         self.dropout_rate = dropout_rate
         self.model_name = model_name
         self.regularizer = regularizer
         
         self._build_VAE()

    def _build_VAE(self):
    
        # Encoder

        encoder_input = Input(shape=self.input_shape, name = 'encoder_input')

        x = encoder_input

        if not self.dropout_rate == 0:
            x = Dropout(self.dropout_rate)(x)

        for i, nnodes in enumerate(self.layer_shapes_encoder):
            x = Dense(units=nnodes)(x)
            x = LeakyReLU(alpha=0.1)(x)
        
        self.mu = Dense(self.latent_space_shape, name='mu')(x)
        self.log_var = Dense(self.latent_space_shape, name='log_var')(x)
    
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon    
    
        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])
        self.mumodel = Model(encoder_input, self.mu)
        self.logmodel = Model(encoder_input, self.log_var)
        self.encoder = Model(encoder_input, encoder_output)

        # Decoderrrr

        decoder_input = Input(shape=(self.latent_space_shape))

        x = decoder_input

        if not self.dropout_rate == 0:
            l_dec = Dropout(self.dropout_rate)(x)

        for i, nnodes in enumerate(self.layer_shapes_decoder):
            x = Dense(units=nnodes, kernel_regularizer=regularizers.L1(self.regularizer))(x)
            x = LeakyReLU(alpha=0.1)(x)

        x = Dense(units=self.input_shape, kernel_regularizer=regularizers.L1(self.regularizer))(x)
       
        #x = sigmoid(x)
        decoder_output = x
        
        self.decoder = Model(decoder_input, decoder_output)

        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
    
        


    def compile(self, learning_rate, r_loss_factor):
            self.learning_rate = learning_rate

            ### COMPILATION
            def vae_r_loss(y_true, y_pred):
                r_loss = K.mean(K.square(y_true - y_pred), axis = 1)
                return r_loss_factor * r_loss

            def vae_kl_loss(y_true, y_pred):
               kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
               return kl_loss

            def vae_loss(y_true, y_pred):
                r_loss = vae_r_loss(y_true, y_pred)
                kl_loss = vae_kl_loss(y_true, y_pred)
                return  r_loss + kl_loss

            optimizer = Adam(lr=learning_rate)
            self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
            
            
    
    
    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_shape
                , self.layer_shapes_encoder
                , self.layer_shapes_decoder
                , self.latent_space_shape
                , self.dropout_rate
                , self.model_name
                , self.regularizer
                ], f)

        self.model.save_weights(os.path.join(folder,'weights/weights.h5'))
        
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
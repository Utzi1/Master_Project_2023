#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:20:38 2022

@author: dirk
"""

import pickle
import os
#import VAE_builder


def load_model(folder):
    
    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = VAE_class(*params)

    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model

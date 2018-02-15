#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:14:52 2018

@author: hagen
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from ase.db import connect
from ase.visualize import mlab
from ase.visualize import view
from ase.io import read
import copy

from dtnn.models import DTNN


model_dir='output2/DTNN_30_60_3_20.0_split_1'
split_dir='output2/split_1'

atom0 = read('conversion/5779.xyz')
r_vec = atom0.positions[1] - atom0.positions[16]

features = {
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'cell': np.eye(3).astype(np.float32),
    'pbc': np.zeros((3,)).astype(np.int64)
}

model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['y']
#%%
fac = np.linspace(0,2,100)
u = []
for a in range(10):
    atom0.positions[16] += .1*r_vec
#    print(atom0.positions)
    u.append(predict(atom0))

#%%
def predict(X):
    with tf.Session() as sess:
        model.restore(sess)
        atom0
        feed_dict = {
        features['numbers']:
            np.array(X.numbers).astype(np.int64),
        features['positions']:
            np.array(X.positions).astype(np.float32)
            }   
        U0_p = sess.run(y, feed_dict=feed_dict)
    return U0_p

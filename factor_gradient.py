#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:14:52 2018

@author: hagen
"""

import argparse
import os

import numpy as np
from numpy import newaxis as na
import tensorflow as tf
import matplotlib.pyplot as plt
from ase.io import read
import copy
from dtnn.models import DTNN

#%%
model_dir='output/DTNN_64_64_3_20.0_split_1'
split_dir='output/split_1'

## Preparing vectors for atom shift
molecule0 = read('conversion/5779.xyz')
r_vec = molecule0.positions[1] - molecule0.positions[14] # shift vector
r_vec /= np.linalg.norm(r_vec)



#%%
features = {
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
    'line_fac': tf.placeholder(tf.float32, shape=(1,1)),
    'line_vec': tf.placeholder(tf.float32, shape=(1,3)),
    'atom_nr' : np.asarray(16, dtype=np.int32),
    'cell': np.eye(3).astype(np.float32),
    'pbc': np.zeros((3,)).astype(np.int64)
}

model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['y']
    
#%%

with tf.Session() as sess:
    model.restore(sess)
    g = tf.get_default_graph()
    with g.gradient_override_map({"Tile": "TileDense"}):
        U0_p = []
        molecule_copy = molecule0.copy()
        feed_dict = {
            features['numbers']:
                np.array(molecule_copy.numbers).astype(np.int64),
            features['positions']:
                np.array(molecule_copy.positions).astype(np.float32),
            features['line_vec']:
                np.array(r_vec[:,na].T).astype(np.float32),
            features['line_fac']:
                np.array([1e-3])[:,na].astype(np.float32)
                }  
            
        for n in range(20):
            U0_p.append(sess.run(y, feed_dict=feed_dict))
            ret = tf.gradients(y, features['line_fac'])
            ret = ret[0].eval(session=sess, feed_dict=feed_dict)
            feed_dict[features['line_fac']] += ret
            print(ret, U0_p[-1])
        
        
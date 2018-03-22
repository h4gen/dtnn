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
from plot_molecule import plot_mol
from scale import scale_matrix, clip_gradients

#%%
model_dir='output/DTNN_64_64_3_20.0_split_1'
split_dir='output/split_1'
molecule0 = read('conversion/5779.xyz')



#%%
features = {
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
    'line_fac': tf.placeholder(tf.float32, shape=(1,1)),
    'line_vec': tf.placeholder(tf.float32, shape=(1,3)),
    'atom_nr' : np.asarray(16, dtype=np.int32),
    'cell': np.eye(3).astype(np.float32),
    'pbc': np.zeros((3,)).astype(np.int64),
    'zmask' : tf.placeholder(tf.float32, shape=(None,)),
    'Hmix' : tf.placeholder(tf.float32, shape=(19,1)),
    'Cmix' : tf.placeholder(tf.float32, shape=(19,1)),
    'Omix' : tf.placeholder(tf.float32, shape=(19,1)),
}

model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['y']
   
#%%
#atoms = list(set(molecule0.numbers))
#ZZ = np.zeros((19,20))
#ZZ[:,atoms] = np.random.normal(size=(19,3))

Hmix = np.zeros(shape=(19,1))
Hmix[np.where( molecule0.numbers == 1)] = 1
Cmix = np.zeros(shape=(19,1))
Cmix[np.where( molecule0.numbers == 6)] = 1
Omix = np.zeros(shape=(19,1))
Omix[np.where( molecule0.numbers == 8)] = 1

 
#%%
P_eta = 1e-5
M_eta = 1e-5
with tf.Session() as sess:
    model.restore(sess)
    g = tf.get_default_graph()
    U0_p = []
    molecule_copy = molecule0.copy()
    feed_dict = {
        features['numbers']:
            np.array(molecule_copy.numbers).astype(np.int64),
        features['positions']:
            np.array(molecule_copy.positions).astype(np.float32),
        features['zmask']:
            np.array((molecule_copy.numbers > 0).astype(np.float32)),
        features['Hmix']:
            Hmix,
        features['Cmix']:
            Cmix,
        features['Omix']:
            Omix
            }  
        
    for n in range(100):
        U0_p.append(sess.run(y, feed_dict=feed_dict))
#        retCC = tf.gradients(tf.reduce_sum(y), [model.fCC])
        ret = tf.gradients(tf.reduce_sum(y), [features['positions']])
        Pgrad = -ret[0].eval(session=sess, feed_dict=feed_dict)
        feed_dict[features['positions']] += P_eta * Pgrad
        
        ret = tf.gradients(tf.reduce_sum(y), [features['Hmix'],features['Cmix'], features['Omix']])
        Hgrad = -ret[0].eval(session=sess, feed_dict=feed_dict)
        Cgrad = -ret[1].eval(session=sess, feed_dict=feed_dict)
        Ograd = -ret[2].eval(session=sess, feed_dict=feed_dict)
        A = np.hstack((feed_dict[features['Hmix']],feed_dict[features['Cmix']],feed_dict[features['Omix']]))
#        print(A)
        A += M_eta * np.hstack(( Hgrad, Cgrad, Ograd))
        A[A<0]=0
        A = scale_matrix(A, np.array([10,7,2]), np.ones_like(feed_dict[features['Hmix']]))
        feed_dict[features['Hmix']] = A[:,0][:,na]
        feed_dict[features['Cmix']] = A[:,1][:,na]
        feed_dict[features['Omix']] = A[:,2][:,na]
#        print([Hgrad,Cgrad,Ograd])
        
        print(U0_p[-1])
#        print(Pgrad)

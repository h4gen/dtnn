#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:24:49 2018

@author: hagen
"""

import os

import numpy as np
from numpy import newaxis as na
import tensorflow as tf
import matplotlib.pyplot as plt
from ase.io import read
from ase import Atoms
import copy
from dtnn.models import DTNN
from sklearn.decomposition import PCA
#from plot_molecule import plot_mol
from scale import scale_matrix, save_mol2, L1_Schedule
import glob
import random
import datetime
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


#%%
model_dir='output_iso17/DTNN_64_64_3_20.0_split_1'
#split_dir='output/split_1'
timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H%M")

isomers = glob.glob('conversion/*.xyz')
#all_pos = np.zeros((len(isomers),19,3))
#for i, iso in enumerate(isomers):
#    rand_mol = random.choice(isomers)
#    molecule0 = read(rand_mol)
#    ids = molecule0.numbers.argsort()
##    molecule0.numbers = molecule0.numbers[ids]
#    all_pos[i] = molecule0.positions[ids]
    



#%%
features = {
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
#    'line_fac': tf.placeholder(tf.float32, shape=(1,1)),
#    'line_vec': tf.placeholder(tf.float32, shape=(1,3)),
#    'atom_nr' : np.asarray(16, dtype=np.int32),
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

with tf.Session() as sess:
    model.restore(sess)
    U0_p = []
    for iso in isomers:
        molecule_copy = read(iso)
        
        Hmix = np.zeros(shape=(19,1))
        Hmix[np.where( molecule_copy.numbers == 1)] = 1
        Cmix = np.zeros(shape=(19,1))
        Cmix[np.where( molecule_copy.numbers == 6)] = 1
        Omix = np.zeros(shape=(19,1))
        Omix[np.where( molecule_copy.numbers == 8)] = 1
        feed_dict = {
        features['numbers']:
            np.array(molecule_copy.numbers).astype(np.int64),
        features['positions']:
            np.array(molecule_copy.positions).astype(np.float32),
#        features['line_vec']:
#            np.array(r_vec[:,na].T).astype(np.float32),
#        features['line_fac']:
#            np.array([fac])[:,na].astype(np.float32),
        features['zmask']:
            np.array((molecule_copy.numbers > 0).astype(np.float32)),
        features['Hmix']:
            Hmix,
        features['Cmix']:
            Cmix,
        features['Omix']:
            Omix
            }  
        U0_p.append(sess.run(y, feed_dict=feed_dict))

isomers[np.argsort(np.array(U0_p).ravel())]

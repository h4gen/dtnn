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
    'pbc': np.zeros((3,)).astype(np.int64),
    'fade' : np.asarray(0, dtype=np.int32),
    'atom_type_nr' : np.asarray(0, dtype=np.int32)
}

model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['y']

#%%    
alpha = .1
beta = .4
gamma =.5
mix_vec = np.zeros(molecule0.numbers.max())
mix_vec[np.array(list(set(molecule0.numbers)))-1] = [alpha, beta, gamma]
#%%
facs = np.linspace(0,4,10)
energies = get_shifted_atom_energies(model, molecule0, 16, facs, r_vec)
plt.plot(facs, energies)

#%%
from scipy.signal import argrelextrema

atom_nr = 16
min_energie_facs = facs[argrelextrema( energies, np.less)]
for fac in min_energie_facs:
    molecule_copy = molecule0.copy()
    molecule_copy.positions[atom_nr] +=  fac * r_vec
    molecule_copy.write('pred_mol_fac' + str(fac) + '.xyz')
    from subprocess import call
    call(['babel', '-ixyz', 'pred_mol_fac' + str(fac) + '.xyz', '-omol2', 'pred_mol_fac' + str(fac) + '.mol2'])


#%%
def get_shifted_atom_energies(model, molecule, atom_nr, linspace, r_vec):
    with tf.Session() as sess:
        model.restore(sess)
        U0_p = []
        molecule_copy = molecule.copy()
        for fac in linspace:
            
            molecule_copy.positions[atom_nr] = molecule.positions[atom_nr] + fac * r_vec
            
            feed_dict = {
            features['numbers']:
                np.array(molecule_copy.numbers).astype(np.int64),
            features['positions']:
                np.array(molecule_copy.positions).astype(np.float32),
            features['line_vec']:
                np.array(r_vec[:,na].T).astype(np.float32),
            features['line_fac']:
                np.array([0])[:,na].astype(np.float32)
                }  
            print(feed_dict[features['positions']][atom_nr])
            U0_p.append(sess.run(y, feed_dict=feed_dict))
    return np.asarray(U0_p).ravel()



#        g = tf.get_default_graph()
#        with g.gradient_override_map({"Tile": "TileDense"}):
#            ret = tf.gradients(y, features['positions'])
#        print(ret)
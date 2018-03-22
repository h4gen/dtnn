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
    'zmask' : tf.placeholder(tf.float32, shape=(None,)),
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
Hmix = np.zeros(shape=(19,1))
Hmix[np.where( molecule0.numbers == 1)] = 1
Cmix = np.zeros(shape=(19,1))
Cmix[np.where( molecule0.numbers == 6)] = 1
Omix = np.zeros(shape=(19,1))
Omix[np.where( molecule0.numbers == 8)] = 1
#%%
facs = np.linspace(0,4,100)
energies = get_shifted_atom_energies(model, molecule0, 16, facs, r_vec)
plt.plot(facs, energies)

#%%
def get_shifted_atom_energies(model, molecule, atom_nr, linspace, r_vec):
    with tf.Session() as sess:
        model.restore(sess)
        U0_p = []
        molecule_copy = molecule.copy()
        for fac in linspace:

#            molecule_copy.positions[atom_nr] = molecule.positions[atom_nr] + fac * r_vec

            feed_dict = {
            features['numbers']:
                np.array(molecule_copy.numbers).astype(np.int64),
            features['positions']:
                np.array(molecule_copy.positions).astype(np.float32),
            features['line_vec']:
                np.array(r_vec[:,na].T).astype(np.float32),
            features['line_fac']:
<<<<<<< HEAD
                np.array([fac])[:,na].astype(np.float32),
            features['zmask']:
                np.array((molecule0.numbers > 0).astype(np.float32)),
            features['Hmix']:
                Hmix,
            features['Cmix']:
                Cmix,
            features['Omix']:
                Omix
=======
                np.array([0])[:,na].astype(np.float32),
            features['zmask']:
                np.ones(19).astype(np.float32),
>>>>>>> master
                }
            U0_p.append(sess.run(y, feed_dict=feed_dict))
#            ret = tf.gradients(y, features['line_fac'])[0]
#            linegrad = -ret.eval(session=sess, feed_dict=feed_dict)
#            retDB = tf.gradients(tf.reduce_sum(y), [model.debug])[0]
#            dbgrad = -retDB.eval(session=sess, feed_dict=feed_dict)
#            dbeval = model.debug.eval(session=sess, feed_dict=feed_dict)
#            print(dbeval)
    return np.asarray(U0_p).ravel()

#%%
feed_dict = {
        features['numbers']:
            np.array(molecule0.numbers).astype(np.int64),
        features['positions']:
            np.array(molecule0.positions).astype(np.float32),
        features['line_vec']:
            np.array(r_vec[:,na].T).astype(np.float32),
        features['line_fac']:
            np.array([1])[:,na].astype(np.float32),
        features['zmask']:
            np.array((molecule0.numbers > 0).astype(np.float32)),
        features['Hmix']:
            Hmix,
        features['Cmix']:
            Cmix,
        features['Omix']:
            Omix
            }
with tf.Session() as sess:
#    sess.run(y, feed_dict=feed_dict)
#    ret = tf.gradients(y, features['line_fac'])[0]
    linegrad = model.debug.eval(session=sess, feed_dict=feed_dict)
    print()

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

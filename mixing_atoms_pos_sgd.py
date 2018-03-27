#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:14:52 2018
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
#from plot_molecule import plot_mol
from scale import scale_matrix, save_mol2
import glob
import random
import datetime
import pickle


#%%
model_dir='output/DTNN_64_64_3_20.0_split_1'
split_dir='output/split_1'
timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H%M")

isomers = glob.glob('conversion/*.xyz')
rand_mol = random.choice(isomers)
molecule0 = read(rand_mol)
ids = molecule0.numbers.argsort()
molecule0.numbers = molecule0.numbers[ids]
molecule0.positions = molecule0.positions[ids]



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

meta_mol_mix = np.random.uniform( size=(19,3))
meta_mol_mix = scale_matrix(meta_mol_mix, np.array([10,7,2]), np.ones((19,1)))
meta_mol_pos = np.random.uniform(low=-3, high=3, size=molecule0.positions.shape)
col_nums = np.array([1,6,8])
meta_mol = Atoms( positions=meta_mol_pos, numbers=(meta_mol_mix * col_nums.T).sum( axis=1).round() )
save_mol2(meta_mol, 'test')
#%%
P_eta = 1e-3
M_eta = 1e-3
with tf.Session() as sess:
    model.restore(sess)
    g = tf.get_default_graph()
    U0_p = []
    for n in range(1000000):
        
            

        rand_mol_path = random.choice(isomers)
        rand_mol = read(rand_mol_path)
        ids = rand_mol.numbers.argsort()
        rand_mol.numbers = rand_mol.numbers[ids]
        rand_mol.positions = rand_mol.positions[ids]
        
        feed_dict0 = {
            features['numbers']:
                np.array(rand_mol.numbers).astype(np.int64),
            features['positions']:
                np.array(meta_mol_pos).astype(np.float32),
            features['zmask']:
                np.array((rand_mol.numbers > 0).astype(np.float32)),
            features['Hmix']:
                Hmix,
            features['Cmix']:
                Cmix,
            features['Omix']:
                Omix
                }  
        
        ret = tf.gradients(tf.reduce_sum(y), [features['positions']])
        Pgrad = -ret[0].eval(session=sess, feed_dict=feed_dict0)
        meta_mol_pos += P_eta * Pgrad
        
        ret = tf.gradients(tf.reduce_sum(y), [features['Hmix'],features['Cmix'], features['Omix']])
        Hgrad = -ret[0].eval(session=sess, feed_dict=feed_dict0)
        Cgrad = -ret[1].eval(session=sess, feed_dict=feed_dict0)
        Ograd = -ret[2].eval(session=sess, feed_dict=feed_dict0)

        meta_mol_mix += M_eta * np.hstack(( Hgrad, Cgrad, Ograd))
        meta_mol_mix[meta_mol_mix<0]=1e-6
        meta_mol_mix = scale_matrix(meta_mol_mix, np.array([10,7,2]), np.ones((19,1)))
      
        feed_dict_meta = {
            features['numbers']:
                np.array(rand_mol.numbers).astype(np.int64),
            features['positions']:
                np.array(meta_mol_pos).astype(np.float32),
            features['zmask']:
                np.array((rand_mol.numbers > 0).astype(np.float32)),
            features['Hmix']:
                meta_mol_mix[:,0,na],
            features['Cmix']:
                meta_mol_mix[:,1,na],
            features['Omix']:
                meta_mol_mix[:,2,na]
                }
                    
        U0_p.append(sess.run(y, feed_dict=feed_dict_meta))

        if n % 100 == 0 :
            meta_mol = Atoms( positions=meta_mol_pos, numbers=(meta_mol_mix * col_nums.T).sum( axis=1).round() )
            directory = timestamp + '_P_eta_' + str(P_eta) + '_M_eta_' + str(M_eta)
            if not os.path.exists(directory):
                os.makedirs(directory)

            meta_mol.write(directory + '/' + 'meta_mol'+str(n)+'.xyz')
#                save_mol2(meta_mol, 'meta_mol' +str(n))
            plt.figure()
            plt.plot( np.array( U0_p).ravel())
            plt.savefig(directory + '/' + str(n))
            plt.show()
            with open(directory + '/' + 'energies.pkl', 'wb') as file:
                pickle.dump(U0_p, file)

        print(n, U0_p[-1])




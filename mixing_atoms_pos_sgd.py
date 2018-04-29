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
from sklearn.decomposition import PCA
#from plot_molecule import plot_mol
from scale import scale_matrix, save_mol2, L1_Schedule, to_proba_isospace, fix_positions, max_likely_numbers
import glob

import datetime
import pickle


#%%
model_dir='output/DTNN_64_64_3_20.0_split_1'
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

molecule0 = read('conversion/4928.xyz')
#all_pos.mean(axis=0)
pca = PCA(n_components=3, whiten=False)
ids = np.argsort(molecule0.numbers)
molecule0.numbers = molecule0.numbers[ids]
molecule0.positions = molecule0.positions[ids]
Hmix = np.zeros(shape=(19,1))
Hmix[np.where( molecule0.numbers == 1)] = 1
Cmix = np.zeros(shape=(19,1))
Cmix[np.where( molecule0.numbers == 6)] = 1
Omix = np.zeros(shape=(19,1))
Omix[np.where( molecule0.numbers == 8)] = 1

meta_mol_mix = np.hstack((Hmix,Cmix,Omix))
meta_mol_mix_isospace, meta_mol_numbers_isospace = to_proba_isospace(meta_mol_mix.copy())
meta_mol_pos = molecule0.positions 
meta_mol = Atoms( positions=meta_mol_pos, numbers=meta_mol_numbers_isospace )
meta_mol.positions = pca.fit_transform(meta_mol.positions)
save_mol2(meta_mol, 'test')


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
#l1_list = [tf.concat((features['Hmix'][i], features['Cmix'][i], features['Omix'][i]), axis=0) for i in range(19)]
#%%
P_eta = 1e-3
M_eta = 0e-3
cut = 4e3
div = 2e4
scale = 1
l1s = L1_Schedule(cut = cut, div= div, scale=scale) 
MAX_ATOM_DIST = 1.5
MIN_ATOM_DIST = .9
INFO = '_L0.5_Norm_Train_iso17_2'
with tf.Session() as sess:
    model.restore(sess)
    g = tf.get_default_graph()
    U0_p = [None]
    l1scale = tf.Variable(1e-6)
    ret = tf.gradients(tf.add(\
                        tf.reduce_sum(y), \
                        tf.multiply(l1scale, \
                        tf.square(\
                        tf.reduce_sum(\
                        tf.sqrt(tf.concat((features['Hmix'],features['Cmix'], features['Omix']), axis=0)))))),\
                       [features['Hmix'],features['Cmix'], features['Omix'], features['positions']])
    for n in range(1000000):
        
        if n % 1000 == 0 :
            directory = timestamp + '_P_eta_' + str(P_eta) + '_M_eta_' + str(M_eta) + '_Atomdist_' + str(MAX_ATOM_DIST) + '_' + str(MIN_ATOM_DIST) + INFO + 'cut' +str(cut) + '_div' +str(div) + '_scale' + str(scale)
            if not os.path.exists(directory):
                os.makedirs(directory)

#            meta_mol.write(directory + '/' + 'meta_mol'+str(n)+'.xyz')
            save_mol2(meta_mol, directory + '/' +  'meta_mol' +str(n))
            if n > 0:
                print('l1 scale',l1scale.eval(session=sess)) 
                plt.figure()
                plt.plot( np.array( U0_p).ravel())
                plt.savefig(directory + '/' + str(n))
                plt.show()
                with open(directory + '/' + 'energies.pkl', 'wb') as file:
                    pickle.dump(U0_p, file)
            
            plt.imshow(meta_mol_mix)
            plt.show()
#            plt.imshow(meta_mol_mix_isospace)
#            plt.show()
            print(meta_mol_mix.sum(axis=0))
            print(meta_mol_mix.sum(axis=1))
            print(meta_mol.numbers)
            


#        rand_mol_path = random.choice(isomers)
#        rand_mol = read(rand_mol_path)
#        ids = rand_mol.numbers.argsort()
#        rand_mol.numbers = rand_mol.numbers[ids]
#        rand_mol.positions = pca.fit_transform(rand_mol.positions[ids])
        
        feed_dict0 = {
            features['numbers']:
                np.array(molecule0.numbers).astype(np.int64),
            features['positions']:
                np.array(meta_mol_pos).astype(np.float32),
            features['zmask']:
                np.array((molecule0.numbers > 0).astype(np.float32)),
            features['Hmix']:
                meta_mol_mix[:,0,na],
            features['Cmix']:
                meta_mol_mix[:,1,na],
            features['Omix']:
                meta_mol_mix[:,2,na]
                }  
        
        
        l1scale.assign(l1s.f(n)).op.run(session=sess, feed_dict=feed_dict0)
#        Hgrad = -ret[0].eval(session=sess, feed_dict=feed_dict0)
#        Cgrad = -ret[1].eval(session=sess, feed_dict=feed_dict0)
#        Ograd = -ret[2].eval(session=sess, feed_dict=feed_dict0)
        Pgrad = -ret[3].eval(session=sess, feed_dict=feed_dict0)
        meta_mol_pos += P_eta * Pgrad
        meta_mol_pos = fix_positions(meta_mol_pos, min_dist=MIN_ATOM_DIST, max_dist=MAX_ATOM_DIST)

#        meta_mol_mix += M_eta * np.hstack(( Hgrad, Cgrad, Ograd))
#        meta_mol_mix[meta_mol_mix<0]=1e-7
#        meta_mol_mix[meta_mol_mix>1]=1
#        meta_mol_mix = scale_matrix(meta_mol_mix, np.array([10,7,2]), np.ones((19,1)))
#        
        meta_mol_numbers_isospace = max_likely_numbers(meta_mol_mix.copy())

        meta_mol = Atoms( positions=meta_mol_pos, numbers=meta_mol_numbers_isospace )

#        l1apply.op.run(session=sess, feed_dict=feed_dict0)
        
        feed_dict_meta = {
            features['numbers']:
                np.array(meta_mol.numbers).astype(np.int64),
            features['positions']:
                np.array(meta_mol.positions).astype(np.float32),
            features['zmask']:
                np.array((meta_mol.numbers > 0).astype(np.float32)),
            features['Hmix']:
                meta_mol_mix[:,0,na],
            features['Cmix']:
                meta_mol_mix[:,1,na],
            features['Omix']:
                meta_mol_mix[:,2,na]
                }
                    
        U0_p.append(sess.run(y, feed_dict=feed_dict_meta))
        

        if n % 100 == 0 :
            print(n, U0_p[-1])




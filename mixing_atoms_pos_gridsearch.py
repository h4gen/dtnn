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
from ase.io import read
from ase import Atoms
from dtnn.models import DTNN
from sklearn.decomposition import PCA
#from plot_molecule import plot_mol
from scale import scale_matrix, save_mol2, L1_Schedule, to_proba_isospace, fix_positions, max_likely_numbers
from sklearn.model_selection import ParameterGrid
import pickle
from tqdm import tqdm
from random import shuffle


#%%
model_dir='output_iso17_2/DTNN_64_64_3_20.0_split_1'
#split_dir='output/split_1'

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
DELTA_MEAN = 1e-4
P_eta_list = {'P_eta': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
M_eta_list = {'M_eta': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
cut_list = {'cut': [0, 1e3, 1e4,]}
div_list = {'div': [1e3, 1e4, 1e5]}
scale_list = {'scale': {0.2, 0.4, 0.6, 0.8, 1}}
MAX_ATOM_DIST_list = {'maad': [1.5, 1.7, 1.9, 2.1]}
MIN_ATOM_DIST_list = {'miad': [.3, .5, .7, .9, 1.1]} 
param_grid = {**P_eta_list, ** M_eta_list, **cut_list, **div_list, **scale_list, **MAX_ATOM_DIST_list, ** MIN_ATOM_DIST_list}
params_list = list(ParameterGrid(param_grid))
shuffle(params_list)

#%%

for paramset in tqdm(params_list):
    
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
    meta_mol_pos = molecule0.positions.copy()
    meta_mol = Atoms( positions=meta_mol_pos, numbers=meta_mol_numbers_isospace )
    meta_mol.positions = pca.fit_transform(meta_mol.positions)
    
    P_eta = paramset['P_eta']
    M_eta = paramset['M_eta']
    cut = paramset['cut']
    div = paramset['div']
    scale = paramset['scale']
    MAX_ATOM_DIST = paramset['maad']
    MIN_ATOM_DIST = paramset['miad']
    energies = []
    mixing_matrices = []
    regs = []
    
    print('\nStarting new grid point with ' + str(paramset))
    
    directory = 'gridsearch/' + str(paramset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + '/' + 'params.pkl', 'wb') as file:
        pickle.dump(paramset, file)
        file.close()

    with tf.Session() as sess:
        l1s = L1_Schedule(cut = cut, div= div, scale=scale) 
        model.restore(sess)
        g = tf.get_default_graph()
        U0_p = [None]
        l1scale = tf.Variable(1e-9)
        ret = tf.gradients(tf.add(\
                            tf.reduce_sum(y), \
                            tf.multiply(l1scale, \
                            tf.square(\
                            tf.reduce_sum(\
                            tf.sqrt(tf.concat((features['Hmix'],features['Cmix'], features['Omix']), axis=0)))))),\
                           [features['Hmix'],features['Cmix'], features['Omix'], features['positions']])
        
        for n in range(100000):
            reg = l1s.f(n)
            if n % 1000 == 0 :
                print(n)
    
                meta_mol.write(directory + '/' + str(n)+'.xyz')
#                save_mol2(meta_mol, directory + '/' +  'meta_mol' +str(n))

                with open(directory + '/' + 'energies.pkl', 'wb') as file:
                    pickle.dump(U0_p, file)
                    file.close()
                with open(directory + '/' + 'matrices.pkl', 'wb') as file:
                    pickle.dump(mixing_matrices, file)
                    file.close()
                with open(directory + '/' + 'reg.pkl', 'wb') as file:
                    pickle.dump(regs, file)
                    file.close()

                if n > 2000 and bool((np.abs((np.array(U0_p).ravel()[-1000:-1] - np.array(U0_p).ravel()[-1001:-2]).mean()) < DELTA_MEAN)):
                    print('Model converged. Stopping...')
                    break
                
            
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
            
            
            l1scale.assign(reg).op.run(session=sess, feed_dict=feed_dict0)
            Hgrad = -ret[0].eval(session=sess, feed_dict=feed_dict0)
            Cgrad = -ret[1].eval(session=sess, feed_dict=feed_dict0)
            Ograd = -ret[2].eval(session=sess, feed_dict=feed_dict0)
            Pgrad = -ret[3].eval(session=sess, feed_dict=feed_dict0)
            meta_mol_pos += P_eta * Pgrad
            
            try:
                meta_mol_pos = fix_positions(meta_mol_pos, min_dist=MIN_ATOM_DIST, max_dist=MAX_ATOM_DIST)
                meta_mol_mix += M_eta * np.hstack(( Hgrad, Cgrad, Ograd))
                if np.any(meta_mol_mix == np.nan) or np.any(meta_mol_mix == np.inf):
                    print('Invalid values in mixing matrix. Skipping')
                meta_mol_mix[meta_mol_mix<0]=1e-7
                meta_mol_mix[meta_mol_mix>1]=1
                meta_mol_mix = scale_matrix(meta_mol_mix, np.array([10,7,2]), np.ones((19,1)))
                meta_mol_numbers_isospace = max_likely_numbers(meta_mol_mix.copy())
            except:
                break
    
            meta_mol = Atoms( positions=meta_mol_pos, numbers=meta_mol_numbers_isospace )
    
            
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
            mixing_matrices.append(meta_mol_mix)
            




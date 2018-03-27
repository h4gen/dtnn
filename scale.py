#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:15:04 2018

@author: hagen
"""



import numpy as np
from numpy import newaxis as na
from subprocess import call



np.random.seed(42)
AA = np.abs(np.random.normal(size=(19,3)))

scale_col = np.array([7,10,2])
scale_row = np.ones(AA.shape[0])
A=AA

def scale_matrix(A, scale_col, scale_row):
    while not (np.all(np.isclose(A.sum(axis=0), scale_col, rtol=1e-7)) and np.all(np.isclose(A.sum(axis=1), scale_row, rtol=1e-3))):
        
        colsum = A.sum(axis=0)
        A/= colsum
        A*= scale_col
    #    print(A.sum(axis=0))
        rowsum = A.sum(axis=1)[:,na]
        A/= rowsum
    
#    print(A.sum(axis=1))
#    print(A.sum(axis=0))
    return A           

def clip_gradients(A, grads, scale = .5):
    new_grads = []
    for i, grad in enumerate(grads):
        new_grads.append(.5 * np.abs(A[:,i].min()) * grad/ np.linalg.norm(grad))
    
    return new_grads

def save_mol2(atom, name):
    atom.write(name+'.xyz')
    call(['babel', '-ixyz', name + '.xyz', '-omol2', name +  '.mol2'])

#%%
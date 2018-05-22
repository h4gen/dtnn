#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:15:04 2018

@author: hagen
"""



import numpy as np
from numpy import newaxis as na
from subprocess import call
from scipy.spatial.distance import cdist
import networkx as nx


def scale_matrix(A, scale_col = np.array([10,7,2]), scale_row = np.ones((19,1))):
    count = 0
    while not (np.all(np.isclose(A.sum(axis=0), scale_col, rtol=1e-4)) and np.all(np.isclose(A.sum(axis=1), scale_row, rtol=1e-8))):
        count += 1
        colsum = A.sum(axis=0)
        A/= colsum
        A*= scale_col
        rowsum = A.sum(axis=1)[:,na]
        A/= rowsum
        
        if count> 100000:
            print('Scale. out of patience')
            raise Exception

    return A           

def clip_gradients(A, grads, scale = .5):
    new_grads = []
    for i, grad in enumerate(grads):
        new_grads.append(.5 * np.abs(A[:,i].min()) * grad/ np.linalg.norm(grad))
    
    return new_grads

def save_mol2(atom, name):
    atom.write(name+'.xyz')
    call(['babel', '-ixyz', name + '.xyz', '-omol2', name +  '.mol2'])
    
class L1_Schedule:
    def __init__(self,cut=100000, div =3e5, scale=1):
        self.cut = cut
        self.div = div
        self.scale = scale
    def f(self, x):
        if x <= self.cut:
            return 1e-4
        else:
            arg = (x-self.cut)*self.scale
            return (arg)/(self.div+np.abs(arg))
        

def to_proba_isospace(atom_mix):
    mixx = []
    numbers = np.zeros(19)
    for i, ats in enumerate([10,7,2]):
        mixx.append(set(atom_mix[:,i].argsort()[-ats:]))

    while not (mixx[0].isdisjoint(mixx[1]) == mixx[1].isdisjoint(mixx[2]) == mixx[2].isdisjoint(mixx[0]) == True):
        for i in range(atom_mix.shape[0]):
            atom_mix[i, np.random.choice([0,1,2], p=atom_mix[i])] = 1
    #    atom_mix[atom_mix!=1] = 1e-3
        atom_mix = scale_matrix(atom_mix)
        mixx = []
        for i, ats in enumerate([10,7,2]):
            mixx.append(set(atom_mix[:,i].argsort()[-ats:]))
    for i, nr in enumerate([1,6,8]):
        numbers[list(mixx[i])] = nr
    
    return atom_mix, numbers

#%%

def fix_positions(mol_pos,eta=.01, max_dist=1.7, min_dist=0.9):
    mol_center = mol_pos.mean(axis=0)
    dist = cdist( mol_pos, mol_pos)
    max_adj = dist + np.diag(np.inf*np.ones(dist.shape[0])) < max_dist
    max_graph = nx.Graph(max_adj)
    max_graphs = list(nx.connected_component_subgraphs(max_graph))
    min_adj = dist + np.diag(np.inf*np.ones(dist.shape[0])) < min_dist
    min_graph = nx.Graph(min_adj)
    min_graphs = list(nx.connected_component_subgraphs(min_graph))
    counter = 0
    
    while not len(min_graphs) == mol_pos.shape[0]:
        counter += 1
        for subgraph in min_graphs:
            sub_atoms = list(subgraph.nodes())
            if len(sub_atoms) > 1:
                sub_center = mol_pos[sub_atoms].mean(axis=0)
                r_vec = mol_pos[sub_atoms] - sub_center
                mol_pos[sub_atoms] += eta*r_vec
        
        dist = cdist( mol_pos, mol_pos)
        min_adj = dist + np.diag(np.inf*np.ones(dist.shape[0])) < min_dist
        min_graph = nx.Graph(min_adj)
        min_graphs = list(nx.connected_component_subgraphs(min_graph))
        
        if counter >= 10000:
            raise Exception()
    
    counter = 0
    while not len(max_graphs) == 1:
        counter += 1
        counter 
        for subgraph in max_graphs:
            sub_atoms = list(subgraph.nodes())
            sub_center = mol_pos[sub_atoms].mean(axis=0)
            r_vec = mol_center - sub_center
            mol_pos[sub_atoms] += eta*r_vec
            
        mol_center = mol_pos.mean(axis=0)
        dist = cdist( mol_pos, mol_pos)
        max_adj = dist + np.diag(np.inf*np.ones(dist.shape[0])) < max_dist
        max_graph = nx.Graph(max_adj)
        max_graphs = list(nx.connected_component_subgraphs(max_graph))
        if counter >= 10000:
            raise Exception()

    
    return mol_pos
      
def max_likely_numbers(atom_mix):
    choice = []
    for i in range(atom_mix.shape[0]):
        choice.append(np.random.choice([0,1,2], p=atom_mix[i]))
    while not (choice.count(0)==10 and choice.count(1)==7 and choice.count(2)==2):  
        choice = []
#        print('choice')
        for i in range(atom_mix.shape[0]):
            choice.append(np.random.choice([0,1,2], p=atom_mix[i]))
            
    return list(map(lambda x: 1 if x==0 else 6 if x==1 else 8, choice))









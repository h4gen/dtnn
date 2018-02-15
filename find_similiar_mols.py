#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:42:07 2018

@author: hagen
"""

from rdkit import Chem
import glob, os

conv_dir='conversion/'


filenames = []
for file in os.listdir("conversion"):
    if file.endswith(".mol2"):
        filenames.append(file)
       
#%%
molecules = []
for file in filenames:
    molecules.append(Chem.MolFromMol2File(conv_dir+file))
    
#%%
from rdkit.Chem.Fingerprints import FingerprintMols
fingerpints = [FingerprintMols.FingerprintMol(mol) for mol in molecules]
#%%
from rdkit.DataStructs import FingerprintSimilarity
import numpy as np

simmat = np.zeros(shape=(len(molecules), len(molecules)))

for i in range(len(molecules)):
    for j in range(i + 1, len(molecules)):
        simmat[i,j] = FingerprintSimilarity(fingerpints[i], fingerpints[j])

#%%
import matplotlib.pyplot as plt
plt.imshow(simmat)

#%%
hi_sim_ids0 = np.where( simmat ==1)
for mol1, mol2 in zip(hi_sim_ids0[0],hi_sim_ids0[1]):
    print(filenames[mol1])
    print(filenames[mol2])
    print('=======')
    
#%%
from rdkit.Chem import MACCSkeys
fingerpints2 = [MACCSkeys.GenMACCSKeys(mol) for mol in molecules]
simmat2 = np.zeros(shape=(len(molecules), len(molecules)))

for i in range(len(molecules)):
    for j in range(i + 1, len(molecules)):
        simmat2[i,j] = FingerprintSimilarity(fingerpints2[i], fingerpints2[j])
        
        #%%
import matplotlib.pyplot as plt
plt.imshow(simmat)

#%%
hi_sim_ids1 = np.where( simmat2 == 1)

for mol1, mol2 in zip(hi_sim_ids1[0],hi_sim_ids1[1]):
    print(filenames[mol1])
    print(filenames[mol2])
    print('=======')



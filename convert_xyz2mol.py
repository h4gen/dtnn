#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:58:30 2018

@author: hagen
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from ase.db import connect
import matplotlib.pyplot as plt

from dtnn.models import DTNN

split_dir='output/split_1'
    
#%%
path = os.path.join(split_dir, 'test_live.db')
c = connect(path)
atomrows = list(c.select())
for nr, row in enumerate(atomrows):
    row.toatoms().write('conversion/' + str(nr) + '.xyz')

#%%
from subprocess import call
for nr, row in enumerate(atomrows):
    call(['babel', '-ixyz', 'conversion/' +  str(nr) + '.xyz', '-omol2', 'conversion/' +  str(nr) + '.mol2'])

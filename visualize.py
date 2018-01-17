#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:14:52 2018

@author: hagen
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from ase.db import connect
from ase.visualize import mlab
from ase.visualize import view
import copy

from dtnn.models import DTNN


model_dir='output/DTNN_30_60_3_20.0_split_1'
split_dir='output/split_1'

path = os.path.join(split_dir, 'test_live.db')
c = connect(path)
l = list(c.select())
l[0].toatoms().write('true_pos.xyz')

features = {
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'cell': np.eye(3).astype(np.float32),
    'pbc': np.zeros((3,)).astype(np.int64)
}

model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['positions']

with tf.Session() as sess:
    model.restore(sess)
    testat = l[0].toatoms()
    feed_dict = {
    features['numbers']:
        np.array(testat.numbers).astype(np.int64),
    features['positions']:
        np.array(testat.positions).astype(np.float32)
        }   
    U0_p = sess.run(y, feed_dict=feed_dict)
 
test_at = copy.copy(l[0])
test_at.positions = U0_p[0,:,:]
test_at.toatoms().write('pred_pos.xyz')

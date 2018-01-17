#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:28:41 2018

@author: hagen
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from ase.db import connect
import matplotlib.pyplot as plt

from dtnn.models import DTNN


model_dir='output/DTNN_30_60_3_20.0_split_1'
split_dir='output/split_1'
    
#%%
path = os.path.join(split_dir, 'test.db')
c = connect(os.path.join(split_dir, 'test.db'))
l = list(c.select())
#%%
    
def predict(dbpath, features, sess, y):
    U0 = []
    U0_pred = []
    count = 0
    with connect(dbpath) as conn:
        n_structures = conn.count()
        for row in conn.select():
            U0.append(row['energy_U0'])

            at = row.toatoms()
            feed_dict = {
                features['numbers']:
                    np.array(at.numbers).astype(np.int64),
                features['positions']:
                    np.array(at.positions).astype(np.float32)
            }
            U0_p = sess.run(y, feed_dict=feed_dict)
            U0_pred.append(U0_p)
            if count % 1000 == 0:
                print(str(count) + ' / ' + str(n_structures))
            count += 1
    return U0, U0_pred
#%%

features = {
    'numbers': tf.placeholder(tf.int64, shape=(None,)),
    'positions': tf.placeholder(tf.float32, shape=(None, 3)),
    'cell': np.eye(3).astype(np.float32),
    'pbc': np.zeros((3,)).astype(np.int64)
}

# load model
model = DTNN(model_dir)
model_output = model.get_output(features, is_training=False)
y = model_output['y']

with tf.Session() as sess:
    model.restore(sess)
    
    print('test_live.db')
    U0_live, U0_pred_live = predict(
        os.path.join(split_dir, 'test_live.db'), features, sess, y
    )
    print('test.db')
    U0, U0_pred = predict(
        os.path.join(split_dir, 'test.db'), features, sess, y
    )
    U0 += U0_live
    U0_pred += U0_pred_live
    U0 = np.vstack(U0)
    U0_pred = np.vstack(U0_pred)

    diff = U0 - U0_pred
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    print('MAE: %.3f eV, RMSE: %.3f eV' % (mae, rmse))

xs = np.arange(0, U0_pred.shape[0])
plt.plot(xs, U0_pred)

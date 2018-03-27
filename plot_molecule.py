#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:44:25 2018

@author: hagen
"""

import plotly.plotly as py
import plotly.graph_objs as go
from ase.io import read

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_mol(pos, color):
    fig = plt.figure()        
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pos.T,c=color)







#trace1 = go.Scatter3d(
#    x=molecule0.positions[:,0] ,
#    y=molecule0.positions[:,1],
#    z=molecule0.positions[:,2],
#    mode='markers',
#    marker=dict(
#        size=12,
#        color=molecule0.numbers,                # set color to an array/list of desired values
#        colorscale='YlGnBu',   # choose a colorscale
#        opacity=0.8
#    )
#)
#
#data = [trace1]
#layout = go.Layout(
#    margin=dict(
#        l=0,
#        r=0,
#        b=0,
#        t=0
#    )
#)
#fig = go.Figure(data=data, layout=layout)
#py.plot(fig, filename='3d-scatter-colorscale')
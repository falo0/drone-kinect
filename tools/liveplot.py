#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:02:57 2019

@author: Dolan
"""
from mpl_toolkits.mplot3d import Axes3D #registers 3D projection, otherwise unused
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

## SET UP PLOT ##
# The default is to plot 3d, although plotting only x,y dimensions is less buggy for now
plot2D = False
#plot2D = True


# -- 2D Plot (tested) --
# For now 2d scatter plot as live updates for 3D scatter plot don't work yet
if plot2D:
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [],[]
    sc = ax.scatter(x,y)
    plt.xlim(-0.3,0.3)
    plt.ylim(-0.3,0.3)
else:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Links_Rechts')
    ax.set_ylabel('Entfernung')
    ax.set_zlabel('Unten_Oben')
    axes = plt.gca()
    axes.set_xlim([-0.5,0.5])
    axes.set_ylim([0, 7])
    axes.set_zlim([-0.3,0.3])
    #scat = ax.scatter([], [], [], c='b', marker='o')   
    #scat = ax.scatter([], [], [], c=['r', 'b', 'g'], marker='o')    
    
    cmap = matplotlib.cm.get_cmap('Set1')
    scat = ax.scatter([], [], [], cmap = "blabla", marker='o')
    scat.set_cmap(cmap)
    
    scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
    scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
    scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='blue', marker = 'o')
    ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['Kalman', 'Camera', 'Depth Sensor'], numpoints = 1)
    
    
def update_2dplot(coords):
    if coords is not None:
        sc.set_offsets(-1 * coords[[0,1]]) #only select what is "links-rechts" and "oben-unten" axes and mirror it
        fig.canvas.draw_idle()
        plt.pause(0.01)
        
def update_3dplot(coords, colors):
    #if coords is not None:
    if coords.dtype is np.dtype('float64'): #andernfalls kann es dtype('O') haben und nur None enthalten
        #print("plotting")
        #print(coords)
        #scat._offsets3d needs a list per coordinate dimension, not an np.array!
        # offsets = ([-1 * 2 * coords[:,0]],  [0.5 * coords[:,2]], [-1 * coords[:,1]])
        offsets = (coords[:,0].tolist(), coords[:,2].tolist(), coords[:,1].tolist())
        #print("offsets")
        #print(offsets)
        scat.set_array(np.array([1, 10, 100]))
        scat._offsets3d = offsets
        fig.canvas.draw_idle()
        plt.pause(0.01)

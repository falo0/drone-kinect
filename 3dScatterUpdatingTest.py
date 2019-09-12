#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:27:09 2019

@author: Dolan
"""

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

G=6.67408e-11
msol=1.989e40
mter=5.972e24
au=1.496e11
dt=.0007
N=30
positions=np.random.rand(N,3)
velocities=np.random.randn(N,3)
acc=np.zeros_like(positions)
masses=np.random.rand(N)

def gravforce(m0,m1,pos0,pos1):
    global G
    dx=pos1[0]-pos0[0]
    dy=pos1[1]-pos0[1]
    dz=pos1[2]-pos0[2]
    r=np.sqrt(dx**2+dy**2+dz**2)
    f=-G*m0*m1/r**2
    ratio=f/r
    fx=dx*ratio
    fy=dy*ratio
    fz=dz*ratio
    return fx, fy, fz

fig,ax=plt.subplots(subplot_kw=dict(projection='3d'))
planets=ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='b',marker='o')

def animate(i):
    acc[:,0]=[sum([gravforce(masses[i],masses[j],positions[i],positions[j])[0]/masses[j] for j in range(1,N) if j != i]) for i in range(len(acc))]
    acc[:,1]=[sum([gravforce(masses[i],masses[j],positions[i],positions[j])[1]/masses[j] for j in range(1,N) if j != i]) for i in range(len(acc))]
    acc[:,2]=[sum([gravforce(masses[i],masses[j],positions[i],positions[j])[2]/masses[j] for j in range(1,N) if j != i]) for i in range(len(acc))]

    velocities[:,0]=velocities[:,0]+acc[:,0]*dt
    velocities[:,1]=velocities[:,1]+acc[:,1]*dt
    velocities[:,2]=velocities[:,2]+acc[:,2]*dt        

    positions[:,0]=positions[:,0]+velocities[:,0]*dt
    positions[:,1]=positions[:,1]+velocities[:,1]*dt
    positions[:,2]=positions[:,2]+velocities[:,2]*dt

    planets.set_sizes(masses[:]*20)
    planets._offsets3d = (positions[:,0],positions[:,1],positions[:2])

ani=FuncAnimation(fig,animate,frames=1000,interval=1,blit=False)
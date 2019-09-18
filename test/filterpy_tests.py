#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:51:42 2019

@author: Dolan
"""

from filterpy.kalman import KalmanFilter
### INITIALIZATION ###
#initial state x0
f = KalmanFilter(dim_x=2, dim_z=1)
f.x = np.array([2., 0.]) # position, velocity
                
#state transition matrix (usually called A, here its called F)
f.F = np.array([[1.,1.],
                [0.,1.]])

#measurement function
f.H = np.array([[1.,0.]])

#by default, P is an Identity matrix. That's assumed to be ok as a start value.
f.P
# In case we assume very high process noises, we could increase P0:
#f.P *= 1000.

#measurement noise. Here it's just a scalar because dim_z = 1
f.R = 5


from filterpy.common import Q_discrete_white_noise
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
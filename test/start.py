#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#For Spyder: Swicht Matplot from inline to auto
#%matplotlib auto
#%matplotlib inline

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #registers 3D projection, otherwise unused
import numpy as np
import time

from pykalman import KalmanFilter
import numpy as np


## DEFINE LOCALIZATION ALGORITHMS ##
# They should all return np.array([x, y, z]) in meters

# For webcam, Kinect RGB cam and maybe when it's dark Kinect IR cam:
def localize_aruco(frame, matrix_coefficients, distortion_coefficients):
    # The following setup might be used once outside the function
    # Define Which ArUco Marker(s) We Use (here we use 6 by 6 bits)
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    # Define The Detector Parameters
    parameters =  cv2.aruco.DetectorParameters_create()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # side length of the used ArUco marker: 0.105 meters
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.105, matrix_coefficients, distortion_coefficients)
    if tvec is not None:
        return tvec[0,0] #assuming there is only one marker in the image
    else:
        return None
    
# For Kinect depth sensor
def localize_depth(frame):
    return np.array([42, 42, 42])

# For drone IMU
def localize_imu(frame):
    return np.array([42, 42, 42])



## DEFINE SENSOR FUSION ALGORITHM ##
    
### SET UP KALMAN FILTER ###
# initialization of the state vector x. It contains the the 3 coordinates of a point
x0 = np.array([0,0,0])
x_prev_posterior = x0
# state transition matrix A. Here, the simple assumption is that the location of the
# point doesn't change.
A = np.identity(3)
# define the process covariance Q. That's the covariance of the error of the
# pure state transition model. Our state transition model is expected to have
# a relatively high error variance <=> a lot of noise, since it is often not true
# that the point is not moving as our state transition model claims.
Q = np.array([[0.1**2, 0., 0.],
              [0., 1**2, 0.],
              [0., 0., 0.1**2]])
#This Q lead to the kalman gain sabilizing at I*0.83. A lower kalman gain
#could be better due to the bad aruco localization. Parameter tuning:
Q = Q*0.3
# is leads to a Kalman gain of I*0.65, which seems to be a bit better.

# initialize the covariance matrix of the posterior state estimation error x_k_true - x_k_posterior
# We use some reasonable values. Here we use Q/2, pretending the posterior state
# estimation has half the noise of the prior state estimation which comes from the
# state transition model. P_k will be updated frequently anyway, so the initialization
# is not too important.
P0 = np.array([[0.1**2, 0., 0.],
              [0., 1**2, 0.],
              [0., 0., 0.1**2]])/2
P_prev_posterior = P0

# R is the covariance matrix of the measurement error x_k_true - z_k
# First, have a look of xlim, ylim, zlim of the plot furhter below
# we expect x and z values roughly between -0.3 and 0.3 meters so assuming
# standard deviation (sigma) of 0.05 for the x and z measurement is resonable.
# This means it is expected that in 95% of the measurements, their errors are
# between +-1.96*sigma = +-1.96*0.1, leading to an 95%-confidence-interval length of 0.196 or roughly 0.2 meters.
# std of 0.05 means a var of 0.05^2 = 0.0025
# we expect y values roughly between 0 and 4 meters, so assuming a std of
# 0.5 meters for the y measurement is reasonable, which means a 
# 95%-confidence-inteveral length of the y measurement error to be 1.96 or roughly 2 meters
# std of 0.5 meters means a var of sqrt(0.5)
R = np.array([[0.05**2, 0., 0.],
              [0., 0.5**2, 0.],
              [0., 0., 0.05**2]])  

# simple kalman update
def kalman_update(z_k):
    
    global x_prev_posterior
    global P_prev_posterior
    
    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior)
    P_k_prior = P_prev_posterior + Q
    
    
    ### MEASURMENT UPDATE ###
    K_k = np.matmul(P_k_prior, np.linalg.inv(P_k_prior + R))
    print(K_k)
    
    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - x_k_prior))
    P_k_posterior = np.matmul((np.identity(3) - K_k), P_k_prior)
    print(P_k_posterior)
    
    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior = x_k_posterior
    P_prev_posterior = P_k_posterior
    
    return(x_k_posterior)   

#kalman_estimation expects a list of xyzt localizations of different sensors
def kalman_estimation(xyzt_list):
    if xyzt_list[0] is not None:
        # Measurement vector z_k
        # For now it just takes the xyz of the first xyzt vector that it was given.
        z_k = xyzt_list[0][:3]
        print(z_k)
        kalman_xyz = kalman_update(z_k)
        print(kalman_xyz)
        return kalman_xyz
    else:
        return None


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
    def update_2dplot(coords):
        if coords is not None:
            sc.set_offsets(-1 * coords[[0,1]]) #only select what is "links-rechts" and "oben-unten" axes and mirror it
            fig.canvas.draw_idle()
            plt.pause(0.01)
        
# -- 3D Plot --
if not plot2D:
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
    scat = ax.scatter([], [], [], c=['r', 'b'], marker='o')         
    def update_3dplot(coords, colors):
        #if coords is not None:
        if coords.dtype is np.dtype('float64'): #andernfalls kann es dtype('O') haben und nur None enthalten
            
            #scat._offsets3d needs a list per coordinate dimension, not an np.array!
            # offsets = ([-1 * 2 * coords[:,0]],  [0.5 * coords[:,2]], [-1 * coords[:,1]])
            offsets = (coords[:,0].tolist(), coords[:,2].tolist(), coords[:,1].tolist())
            print("offsets")
            print(offsets)
            scat._offsets3d = offsets
            fig.canvas.draw_idle()
            plt.pause(0.01)



## INITIAL SENSOR SETTINGS ##
#read in webcam calibration data, created in camera_calibration.py
cv_file = cv2.FileStorage("camera_calibration/MBP13Late2015Distortion.yaml", cv2.FILE_STORAGE_READ)
wc_matrix_coefficients = cv_file.getNode("camera_matrix").mat()
wc_distortion_coefficients = cv_file.getNode("dist_coeff").mat()
cv_file.release()


## GET ALL SENSORS REDY ##
wc_cap = cv2.VideoCapture(0)


## REPETITIVE PROCESS OF OBTAINING AND PROCESSING SENSOR DATA ##
verbose = False
while True:
    ## READ ALL SENSORS, GET TIMESTAMPS ##
    # Sensor: WebCam
    wc_ret, wc_frame = wc_cap.read()
    wc_t = time.time()
    # Sensor: Kinect rgb Cam
    kc_frame = 42
    kc_t = time.time()
    # Sensor: Kinect Depth sensor -> kd
    kd_frame = 42
    kd_t = time.time()
    
    
    ## SENSOR READING ERRORS ##
    if not wc_ret:
        if verbose: print("Error: No Image Frame Obtained From The Webcam")
        break
    

    ## GET LOCALIZATIONS ACCORDING TO ERRORS, COMBINE WITH TIMESTAMPS ##
    wc_xyz = localize_aruco(wc_frame, wc_matrix_coefficients, wc_distortion_coefficients)
    if wc_xyz is None:
        if verbose: print("skipping Webcam frame: no translation vector could be obtained in this frame.")
        wc_xyzt = None
    else:
        #w_xyz = np.array([1,2,3])
        #Transform wc_xyz
        # for some reasons, "Links_Rechts"-values only give roughly half the meters and
            # "Entfernung"-values roughly give double the meters. Using 2 and 0.5 as prefactors solves this
            # now the coordinates are very roughly correct in all dimensions, they are given in meters.
            # but its still not perfect, probably the calibration is wrong.
        #[-1 * 2 * coords[:,0]],  [0.5 * coords[:,2]], [-1 * coords[:,1]]
        wc_xyz = np.multiply(wc_xyz, np.array([-2., -1., 0.5]))
        
        #Add time
        wc_xyzt = np.append(wc_xyz, wc_t)
        if verbose: print(wc_xyzt)
    
    #kc_xyz = localize_aruco(kc_frame, kc_matrix_coefficients, kc_distortion_coefficients)
    #kd_xyz = localize_depth(...)    
    
    #When just using Kinect cam and depth sensor: get kc_xyzt, kd_xyzt
    #For DRone IMU, get dr_xyzt
        
        
    ## FUSE ALL LOCALIZATIONS ##
    #example when just using Kinect cam and depth sensor:
    #kalman_estimation([kc_xyzt, kd_xyzt])
    
    #Just using wc_xyzt, which is already enough for a Kalman filter, as it
    #can fuse estimations from past values and current camera observations
    kalman_xyz = kalman_estimation([wc_xyzt])
    
    #For testing multiple sensor inputs
    #Simulating another sensor by transforming wc_xyzt
    if wc_xyz is not None:
        simsens_xyz = 1.2 * wc_xyz
        simsens_xyzt = np.append(simsens_xyz, wc_xyzt[-3])
    else:
        simsens_xyz = None
        simsens_xyzt = None
    
    kalman_xyz = kalman_estimation([wc_xyzt, simsens_xyzt])

    

    ## UPDATE THE PLOT ##
    # Just plot the Kalman estimation:
    if plot2D:
        update_2dplot(kalman_xyz)
    else:
        #coords = np.array([kalman_xyz])
        coords = np.vstack([kalman_xyz, wc_xyz])
        colors = ['r', 'b']
        update_3dplot(np.vstack(coords), colors)
    
    #Plot sensor localizations and kalman estimation - 3 points with differnt colors
    #points = np.vstack([wc_xyz, simsens_xyz, kalman_xyz])
    #update_3dplot(points)
    
    
    
    
## TURN OFF / DISCONNECT ALL SENSORS ##    
wc_cap.release()


#import numpy as np
#a = np.array([[0,1.2], [2,4]])
#a = np.array([[0,0], [0,2]])#
#nonzeros = a[np.nonzero(a)]
#if nonzeros.size != 0:
#    minvalue = np.min(nonzeros)
#    coords = np.where(a == minvalue)
#else:
#    print("object in front of Kinect")

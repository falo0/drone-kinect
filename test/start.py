#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#For Spyder: Swicht Matplot from inline to auto
#%matplotlib auto
#%matplotlib inline

import cv2
import cv2.aruco as aruco
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #registers 3D projection, otherwise unused
import numpy as np
import time

import kalman as kal
import liveplot as lplt


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
    # kalman_xyz expects a specific data input,
    # e.g. measurements = [[kc_xyzt, 'kc'], [kd_zt, 'kd']]

    #measurements = [[wc_xyzt, 'wc']]
    #kalman_xyz = kalman_estimation(measurements)
    
    #kalman_xyz = kalman_estimation([wc_xyz], [wc_t], ['wc'])

    
    
    #For testing multiple sensor inputs
    #Simulating another sensor by transforming wc_xyzt
    
    if wc_xyz is not None:
        sim_z = 1.2 * np.array([wc_xyz[2]]) #sim_z needs to be a np.array!
        sim_t = wc_t
        
        #kalman_xyz = kalman_estimation([sim_z], [sim_t], ['sim'], method = 'velocity')
        #here an example of wc_xyz being none, but sim in theory not none: the filter should still update with sim only.
        #kalman_xyz = kalman_estimation([None, sim_z], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')
        
        kalman_xyz = kal.kalman_estimation([wc_xyz, sim_z], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')

    else:
        sim_z = None
        #simsens_xyzt = None

    # Kalman update with velocities of just wc_xyz for now:
    #kalman_xyz = kalman_estimation([wc_xyz], [wc_t], ['wc'], method = 'velocity')
    
    # Kalman update with velocities of both, wc and sim
    #kalman_xyz = kal.kalman_estimation([wc_xyz, sim_z], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')
    
    # Kalman update with velocities of just sim (happens often, when wc doesn't detect ArUco)
    #kalman_xyz = kalman_estimation([sim_z], [sim_t], ['sim'], method = 'velocity')
    
    #here an example of wc_xyz being none, but sim in theory not none: the filter should still update with sim only.
    #kalman_xyz = kalman_estimation([None, np.array([0.7])], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')

    # wc_xyz valid but sim_z is None
    #kalman_xyz = kalman_estimation([wc_xyz, None], [wc_t, sim_t], ['wc', 'sim'], method = 'steady')
    
    #kalman_xyz = kalman_estimation([None, None], [wc_t, sim_t], ['wc', 'sim'], method = 'steady')

    
    
    
    
##### Funktioniert! Als nächstes sim_z auch noch reinfusionieren!     #######

    ## UPDATE THE PLOT ##
    # Just plot the Kalman estimation:
    if wc_xyz is not None:
        #coords = np.array([kalman_xyz])
        print("Kalman result")
        print(kalman_xyz)
        print(wc_xyz)
        sim_xyz = np.array([0, 0, sim_z])
        
        coords = np.vstack([kalman_xyz, wc_xyz, sim_xyz])
        print(coords)
        colors = ['r', 'b', 'g']
        lplt.update_3dplot(np.vstack(coords), colors)
        
        #or just plot 2d
        #lplt.update_2dplot(kalman_xyz)
    
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

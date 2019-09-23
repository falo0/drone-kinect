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
x_prev_posterior_vel = np.zeros(6)
# state transition matrix A. Here, the simple assumption is that the location of the
# point doesn't change.
A = np.identity(3)
# A_vel is updated every time with the delta t's, since we don't assume constant delta t's

# define the process covariance Q. That's the covariance of the error of the
# pure state transition model. Our state transition model is expected to have
# a relatively high error variance <=> a lot of noise, since it is often not true
# that the point is not moving as our state transition model claims.
Q = np.array([[0.1**2, 0., 0.],
              [0., 1**2, 0.],
              [0., 0., 0.1**2]])
Q_vel = np.zeros((6, 6), float)
Q_vel[0:3, 0:3] = Q
Q_vel[3:6, 3:6] = np.array([[0.05**2, 0., 0.],
                            [0., 0.05**2, 0.],
                            [0., 0., 0.05**2]])

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
P_prev_posterior_vel = np.identity(6) * 0.01

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
              [0., 0., 0.05**2]]) #should be deprecated later

# steady point sensors
R_wc = np.array([[0.05**2, 0., 0.],
              [0., 0.5**2, 0.],
              [0., 0., 0.05**2]]) 
R_kc = np.array([[0.06**2, 0., 0.],
              [0., 0.6**2, 0.],
              [0., 0., 0.06**2]])
R_kd = np.array([0.08**2])
R_sim = np.array([0.08**2]) 

# point sensors extended by indirect velocity measurement
R_wc_vel = np.zeros((6, 6), float)
R_wc_vel[0:3, 0:3] = R_wc
R_wc_vel[3:6, 3:6] = np.array([[0.005**2, 0., 0.],
                            [0., 0.005**2, 0.],
                            [0., 0., 0.005**2]])

R_kc_vel = np.zeros((6, 6), float)
R_kc_vel[0:3, 0:3] = R_kc
R_kc_vel[3:6, 3:6] = np.array([[0.005**2, 0., 0.],
                            [0., 0.005**2, 0.],
                            [0., 0., 0.005**2]])

R_kd_vel = np.diag([R_kd[0], 0.005**2])

R_sim_vel = np.diag([R_sim[0], 0.005**2])
    
# Steady point linear kalman update
def kalman_update_steady(z_k):
    
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
    
# Steady point linear kalman update that supports multiple sensor input
def kalman_update_steady_multiple(z_k, H, R):
    
    global x_prev_posterior
    global P_prev_posterior
    
    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior)
    P_k_prior = P_prev_posterior + Q
    
    
    ### MEASURMENT UPDATE ###
    K_k = np.matmul(np.matmul(P_k_prior, H.transpose()), np.linalg.inv(H @ P_k_prior @ H.transpose() + R))
    print("Kalman Gain:")
    print(K_k)
    
    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - np.matmul(H, x_k_prior)))
    P_k_posterior = np.matmul((np.identity(3) - np.matmul(K_k, H)), P_k_prior)
    print(P_k_posterior)
    
    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior = x_k_posterior
    P_prev_posterior = P_k_posterior
    
    print("kalman estimatoin:")
    print(x_k_posterior)
    
    return(x_k_posterior) 
    
# constant velocity linear kalman update
def kalman_update_velocity(z_k, H, R, A):
    global x_prev_posterior_vel
    global P_prev_posterior_vel
    
    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior_vel)
    P_k_prior = A @ P_prev_posterior_vel @ A.transpose() + Q_vel
    
    
    ### MEASURMENT UPDATE ###
    K_k = np.matmul(np.matmul(P_k_prior, H.transpose()), np.linalg.inv(H @ P_k_prior @ H.transpose() + R))
    print("Kalman Gain:")
    print(K_k)
    
    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - np.matmul(H, x_k_prior)))
    P_k_posterior = np.matmul((np.identity(6) - np.matmul(K_k, H)), P_k_prior)
    print(P_k_posterior)
    
    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior_vel = x_k_posterior
    P_prev_posterior_vel = P_k_posterior
    
    print("kalman estimatoin:")
    print(x_k_posterior)
    
    return(x_k_posterior)

#kalman_estimation expects a list of listes of localizations and identifiers of different sensors
# e.g. update from kinect cam and kinect depth sensor:
# e.g. sensor_readings, sensor_names = [kc_xyzt, kd_zt], ['kc', 'kd']
# old: [[kc_xyzt, 'kc'], ['kc, 'kd']]
# For testing:
sensor_readings = [np.array([0.5, 0.5, 0.5]), np.array([0.4])]
reading_times = [123456, 123456]
sensor_names = ['kc', 'kd']


# Storing values of previous invocations of kalman_estmiation, necessary to
# calculate differences dx, dy, dz, dt and therefore velocities vx, vy, vz
wc_xyz_prev = kc_xyz_prev = np.array([0,0,0])
kd_z_prev = sim_z_prev = np.array([0])
wc_t_prev = kc_t_prev = kd_t_prev = kup_t_prev = sim_t_prev = time.time()
#sim_t_prev = np.array([time.time()])
kup_txtytz = np.ones(3) * time.time()


def kalman_estimation(sensor_readings, reading_times, sensor_names, method = 'steady'):
    
    sensor_num = len(sensor_readings)
    if(sensor_num == 0):
        print("kalman_estimation() must not have empty lists as parameters")
        return None
    if(not (sensor_num == len(reading_times) and len(reading_times) == len(sensor_names))):
        print("all function parameters of kalman_estimation() must be lists of same length")
        return None
    if all(elem is None for elem in sensor_readings):
        print("All sensor readings were None, returning previous or initial Kalman estimation")
        if(method == 'steady'):
            return x_prev_posterior
        if(method == 'velocity'):
            return x_prev_posterior_vel

    #mylist = [np.array([2,1]), None]
    
    
    if (method == 'steady'):
        # Create measurement vector z_k
        #print(sensor_readings)
        
        z_k_list = []
        
        # simple list of the covariance matrices of all sensors
        R_list = []
        
        # simple list of submatrices that make up the H matrix
        H_list = []
        
        valid_sensor_num = 0
        
        for i in range(sensor_num):
            if(sensor_readings[i] is not None):
                valid_sensor_num += 1
                z_k_list.append(sensor_readings[i])
                if(sensor_names[i] == 'wc'):
                    # obtain the sensor's covariance matrix
                    R_list.append(R_wc)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.identity(3)) #only if we ignore velocity
                elif (sensor_names[i] == 'kc'):
                    # obtain the sensor's covariance matrix
                    R_list.append(R_kc)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.identity(3)) #only if we ignore velocity
                elif (sensor_names[i] == 'kd'):
                    # obtain the sensor's covariance matrix
                    R_list.append(R_kd)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.array([0, 0, 1])) #only if we ignore velocity
                elif (sensor_names[i] == 'sim'):
                    # obtain the sensor's covariance matrix
                    R_list.append(R_sim)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.array([0, 0, 1])) #only if we ignore velocity
                else:
                    print("undefined keyword used in sensor_names in kalman_estimation()")
            else:
                print("The sensor reading of sensor " + sensor_names[i] + " was None in this update, updating with the remaining 'not None' sensors")
        
        #Convert z_k_list to the measurement vector (np.array) z_k
        z_k = np.concatenate(z_k_list)
        l = len(z_k)
        print(z_k)
        
        
        #Convert H_list to the real measurement matrix H
        #print(H_list)
        H_comb = np.vstack(H_list)
        print("H")
        print(H_comb)
        #Convert R_list to the real Cov-matrix of all combined Sensors
        # Not generalized example with R_wc and R_sim
        #R_comb = np.zeros((4, 4), float)
        #R_comb[0:3, 0:3] = R_wc
        #R_comb[3:4, 3:4] = R_sim
        #R_list = [R_wc, R_sim]
        
        # Initialize combined sensor covariance matrix
        R_comb = np.zeros((l, l), float)
        
        # Fill it with the covariance matrices of each sensor
        last_element = 0
        for i in range(valid_sensor_num):
            own_length = len(R_list[i])
            minidx = last_element
            maxidx = last_element + own_length
            R_comb[minidx:maxidx, minidx:maxidx] = R_list[i]
            last_element += own_length
        
        print(R_comb)
        
        #print(z_k)
        #kalman_xyz = kalman_update_steady(z_k)
        kalman_xyz = kalman_update_steady_multiple(z_k, H_comb, R_comb)
        print(kalman_xyz)
        return kalman_xyz
    elif (method == 'velocity'):
        
         #global measurements_prev
    #if(len(measurements_prev) == 0):
        # initialize measuerements_prev
    #    measurements_prev = measurements
        global wc_xyz_prev, wc_t_prev, kc_xyz_prev, kc_t_prev, kd_z_prev
        global kd_t_prev, sim_z_prev, sim_t_prev, kup_txtytz
        print(sim_z_prev)
        
        # initialize measurement vector e.g. z_k = np.array([kc_x, kc_y, kc_z, kc_vx, kc_vy, kc_vz, kd_z, kd_vz]) 
        z_k_list = []
        
        # simple list of the covariance matrices of all sensors
        R_list = []
        
        # simple list of submatrices that make up the H matrix
        H_list = []
        
        # initialize A
        A = np.identity(6)
        
        valid_sensor_num = 0
        
        for i in range(sensor_num):
            if(sensor_readings[i] is not None):
                
                valid_sensor_num +=1
                
                if(sensor_names[i] == 'wc'):
                    # calculate velocity
                    xyz_delta = sensor_readings[i] - wc_xyz_prev
                    
                    # time between measurements of the same sensor (I think it's wrong):
                    wc_delta_t = reading_times[i] - wc_t_prev
                    
                    wc_vel = xyz_delta/wc_delta_t
                    
                    # fill measurement vector with measured coordinates and velocity
                    z_k_list.append(sensor_readings[i])
                    z_k_list.append(wc_vel)
                    
                    
                    # relevant in the state transition matrix A is the time between 
                    # the Kalman updates, not the time between measurements of the
                    # same sensor
                    now = time.time()
                    kup_dtxdtydtz = now - kup_txtytz
                    
                    # write delta t's in A
                    A[0:3, 3:6] = np.identity(3) * kup_dtxdtydtz
                    
                    # store measurement for later velocity calculation
                    wc_xyz_prev = sensor_readings[i]
                    wc_t_prev = reading_times[i]
                    kup_txtytz = np.ones(3) * now
                    # obtain the sensor's covariance matrix
                    R_list.append(R_wc_vel)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.identity(6))
                    
             
                elif (sensor_names[i] == 'kc'):
                    # calculate velocity
                    xyz_delta = sensor_readings[i] - kc_xyz_prev
                    
                    # time between measurements of the same sensor (I think it's wrong):
                    kc_delta_t = reading_times[i] - kc_t_prev
                    
                    kc_vel = xyz_delta/kc_delta_t
                    
                    # fill measurement vector with measured coordinates and velocity
                    z_k_list.append(sensor_readings[i])
                    z_k_list.append(kc_vel)
                    
                    
                    # relevant in the state transition matrix A is the time between 
                    # the Kalman updates, not the time between measurements of the
                    # same sensor
                    now = time.time()
                    kup_dtxdtydtz = now - kup_txtytz
                    
                    
                    # write delta t's in A
                    A[0:3, 3:6] = np.identity(3) * kup_dtxdtydtz
                    
                    
                    # store measurement for later velocity calculation
                    kc_xyz_prev = sensor_readings[i]
                    kc_t_prev = reading_times[i]
                    kup_txtytz = np.ones(3) * now
                    # obtain the sensor's covariance matrix
                    R_list.append(R_kc_vel)
                    # define the rows in the H matrix that correspond to the given sensor
                    H_list.append(np.identity(6))
                elif (sensor_names[i] == 'kd'):
                    # calculate velocity
                    z_delta = sensor_readings[i] - kd_z_prev
                    
                    # time between measurements of the same sensor (I think it's wrong):
                    kd_delta_t = reading_times[i] - kd_t_prev
                    
                    kd_vel = z_delta/kd_delta_t
                    
                    # fill measurement vector with measured coordinates and velocity
                    z_k_list.append(sensor_readings[i])
                    z_k_list.append(kd_vel)
                    
                    
                    # relevant in the state transition matrix A is the time between 
                    # the Kalman updates, not the time between measurements of the
                    # same sensor
                    now = time.time()
                    #kup_dtz = now - kup_txtytz[2]
                    
                    kup_dtxdtydtz = now - kup_txtytz
                    
                    # write the delta_t belonging to vz in A, assume 0 velocity for not updated velocities:
                    #A[2, 5] = kup_dtz
                    # New approach according to Fabian, same delta t for all velocities:
                    A[0:3, 3:6] = np.identity(3) * kup_dtxdtydtz
                    
                    
                    # store measurement for later velocity calculation
                    kd_z_prev = sensor_readings[i]
                    kd_t_prev = reading_times[i]
                    #kup_txtytz[2] = now
                    kup_txtytz = now
                    # obtain the sensor's covariance matrix
                    R_list.append(R_kd_vel)
                    # define the rows in the H matrix that correspond to the given sensor
                    # this sensor just updates z and vz:
                    H_list.append(np.array([[0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1]]))
                    
                    
                    
                    
                elif (sensor_names[i] == 'sim'):
                    # calculate velocity
                    z_delta = sensor_readings[i] - sim_z_prev
                    
                    # time between measurements of the same sensor (I think it's wrong):
                    sim_delta_t = reading_times[i] - sim_t_prev
                    
                    #print(sensor_readings[i])
                    #print(sim_z_prev)
                    #print(z_delta)
                    #print(sim_delta_t)
                    
                    sim_vel = z_delta/sim_delta_t
                    
                    #print(sensor_readings[i])
                    #print(z_delta, sim_delta_t)
                    
                    
                    # fill measurement vector with measured coordinates and velocity
                    z_k_list.append(sensor_readings[i])
                    z_k_list.append(sim_vel)
                    #print(z_k_list)
                    
                    # relevant in the state transition matrix A is the time between 
                    # the Kalman updates, not the time between measurements of the
                    # same sensor
                    now = time.time()
                    #kup_dtz = now - kup_txtytz[2]
                    kup_dtxdtydtz = now - kup_txtytz
                    
                    # write the delta_t belonging to vz in A, assume 0 velocity for not updated velocities:
                    #A[2, 5] = kup_dtz
                    # New approach according to Fabian, same delta t for all velocities:
                    A[0:3, 3:6] = np.identity(3) * kup_dtxdtydtz
                    
                    
                    
                    
                    # store measurement for later velocity calculation
                    sim_z_prev = sensor_readings[i]
                    sim_t_prev = reading_times[i]
                    #kup_txtytz[2] = now
                    kup_txtytz = now
                    # obtain the sensor's covariance matrix
                    R_list.append(R_sim_vel)
                    # define the rows in the H matrix that correspond to the given sensor
                    # this sensor just updates z and vz:
                    H_list.append(np.array([[0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1]]))
                else:
                    print("undefined keyword used in sensor_names in kalman_estimation()")
            else:
                print("The sensor reading of sensor " + sensor_names[i] + " was None in this update, updating with the remaining 'not None' sensors")
        
        
        print("A")
        print(A)
        
        #Convert H_list to the real measurement matrix H
        #print(H_list)
        H_comb = np.vstack(H_list)
        print("H")
        print(H_comb)
        #Convert R_list to the real Cov-matrix of all combined Sensors
        # Not generalized example with R_wc and R_sim
        #R_comb = np.zeros((4, 4), float)
        #R_comb[0:3, 0:3] = R_wc
        #R_comb[3:4, 3:4] = R_sim
        #R_list = [R_wc, R_sim]
        
        
        #Convert z_k_list of vectors to the single z_k vector
        print(z_k_list)
        z_k = np.concatenate(z_k_list)
        l = len(z_k)
        print("zk")
        print(z_k)
        
        # Initialize combined sensor covariance matrix
        R_comb = np.zeros((l, l), float)
        
        # Fill it with the covariance matrices of each sensor
        last_element = 0
        for i in range(valid_sensor_num):
            own_length = len(R_list[i])
            minidx = last_element
            maxidx = last_element + own_length
            R_comb[minidx:maxidx, minidx:maxidx] = R_list[i]
            last_element += own_length
        
        print(R_comb)
        
        
        
        #print(z_k)
        #kalman_xyz = kalman_update_steady(z_k)
        kalman_xyzvxvyvz = kalman_update_velocity(z_k, H_comb, R_comb, A)
        kalman_xyz = kalman_xyzvxvyvz[:3] #the velocities stay intern
        print(kalman_xyz)
        return kalman_xyz

    
    
#    if measurements[0][0] is not None:
#        # Measurement vector z_k
#        # For now it just takes the xyz of the first xyzt vector that it was given.
#        z_k = measurements[0][0][:3]
#        t_k = measurements[0][0][3]
#        print(z_k)
#        # extend the measurement vector by velocities
#        meas_diff = measurements - measurements_prev
#        print("maes_diff")
#        print(meas_diff)
#        
#        kalman_xyz = kalman_update_steady(z_k)
#        print(kalman_xyz)
#        
#        #------------------
#        ### INCLUDING VELOCITY INTO STATE- AND MEASUREMENT-VECTOR ###
#        # measurements_prev = measurements
#        kalman_update_velocity(z_k)
#        
#        
#        
#        #------------------
#        
#        return kalman_xyz
#    else:
#        return None


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
    scat = ax.scatter([], [], [], c=['r', 'b', 'g'], marker='o')         
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
    else:
        sim_z = None
        #simsens_xyzt = None

    # Kalman update with velocities of just wc_xyz for now:
    #kalman_xyz = kalman_estimation([wc_xyz], [wc_t], ['wc'], method = 'velocity')
    
    # Kalman update with velocities of both, wc and sim
    kalman_xyz = kalman_estimation([wc_xyz, sim_z], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')
    
    # Kalman update with velocities of just sim (happens often, when wc doesn't detect ArUco)
    #kalman_xyz = kalman_estimation([sim_z], [sim_t], ['sim'], method = 'velocity')
    
    #here an example of wc_xyz being none, but sim in theory not none: the filter should still update with sim only.
    #kalman_xyz = kalman_estimation([None, np.array([0.7])], [wc_t, sim_t], ['wc', 'sim'], method = 'velocity')

    # wc_xyz valid but sim_z is None
    #kalman_xyz = kalman_estimation([wc_xyz, None], [wc_t, sim_t], ['wc', 'sim'], method = 'steady')

    
    
    
    
##### Funktioniert! Als nächstes sim_z auch noch reinfusionieren!     #######

    ## UPDATE THE PLOT ##
    # Just plot the Kalman estimation:
    if wc_xyz is not None:
        if plot2D:
            update_2dplot(kalman_xyz)
        else:
            #coords = np.array([kalman_xyz])
            print("Kalman result")
            print(kalman_xyz)
            print(wc_xyz)
            sim_xyz = np.array([0, 0, sim_z])
            
            coords = np.vstack([kalman_xyz, wc_xyz, sim_xyz])
            print(coords)
            colors = ['r', 'b', 'g']
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

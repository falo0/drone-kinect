#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #registers 3D projection, otherwise unused
import numpy as np
from os import chdir, getcwd

# Read In The Camera Callibration Data, Created In camera_calibration.py
cv_file = cv2.FileStorage("camera_calibration/MBP13Late2015Distortion.yaml", cv2.FILE_STORAGE_READ)
matrix_coefficients = cv_file.getNode("camera_matrix").mat()
distortion_coefficients = cv_file.getNode("dist_coeff").mat()
cv_file.release()
# Define Which ArUco Marker(s) We Use (here we use 6 by 6 bits)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Define The Detector Parameters
parameters =  cv2.aruco.DetectorParameters_create()

#For Spyder: Swicht Matplot from inline to auto
#%matplotlib auto
#%matplotlib inline


#----------------------------------
# Working With Images From Video Stream
# =============================================================================
# cap = cv2.VideoCapture(0)
# 
# for i in range(3):
#     ret, frame = cap.read()
#     
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     
#     parameters =  cv2.aruco.DetectorParameters_create()
#     corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#     frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
#     
#     #cv2.imshow('frame_markers', frame_markers)
#     matplotlib.pyplot.imshow(frame_markers)
#     matplotlib.pyplot.imshow(frame)
#     matplotlib.pyplot.imshow(gray, cmap = "gray")
#     
# 
# cap.release()
# 
# frame.shape
# frame[0,0]
# frame
# corners[0]
# =============================================================================
#----------------------------------


# Working With Stored Images For Debugging
tvecs = np.empty((0,3))
fnames = []
chdir('../example_pictures')
for fname in glob.glob('*.jpg'):
    fnames.append(fname)
    
    frame = cv2.imread(fname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
    #cv2.imshow('frame_markers', frame_markers)
    matplotlib.pyplot.imshow(frame_markers)
    
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.07, matrix_coefficients, distortion_coefficients)
    tvecs = np.concatenate((tvecs, tvec[0]))
    #print(tvec)
chdir('../drone-kinect')  
print(fnames)
print(tvecs)


# Plotting the translation vectors (location in space) from the pose
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot([1.], [1.], [1.], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
ax.scatter(tvecs[:,0], tvecs[:,2], -1 * tvecs[:,1], marker='o')
for i in range(tvecs.shape[0]):
    ax.text(tvecs[i,0], tvecs[i,2], -1 * tvecs[i,1],  fnames[i] , size=10, zorder=1, color='k') 
ax.set_xlabel('Links_Rechts')
ax.set_ylabel('Entfernung')
ax.set_zlabel('Unten_Oben')
plt.show()














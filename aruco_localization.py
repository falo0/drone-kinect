#!/usr/bin/env python3

#For Spyder: Swicht Matplot from inline to auto
#%matplotlib auto
#%matplotlib inline

import cv2
import cv2.aruco as aruco
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #registers 3D projection, otherwise unused
import numpy as np
from os import chdir
import time







## DEFINE LOCALIZATION ALGORITHMS ##
# They should all return np.array([x, y, z]) in meters

# For webcam, Kinect RGB cam and maybe when it's dark Kinect IR cam:
def localize_aruco(frame, matrix_coefficients, distortion_coefficients):
    # The following setup might be used once outside the function
    # Define Which ArUco Marker(s) We Use (here we use 6 by 6 bits)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Define The Detector Parameters
    parameters =  cv2.aruco.DetectorParameters_create()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.07, matrix_coefficients, distortion_coefficients)
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



## SET UP PLOT ##
# For now 2d scatter plot as live updates for 3D scatter plot don't work yet
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(-0.3,0.3)
plt.ylim(-0.3,0.3)
#plt.draw()


## INITIAL SENSOR SETTINGS ##
#read in webcam calibration data, created in camera_calibration.py
cv_file = cv2.FileStorage("camera_calibration/MBP13Late2015Distortion.yaml", cv2.FILE_STORAGE_READ)
wc_matrix_coefficients = cv_file.getNode("camera_matrix").mat()
wc_distortion_coefficients = cv_file.getNode("dist_coeff").mat()
cv_file.release()


## GET ALL SENSORS REDY ##
wc_cap = cv2.VideoCapture(0)


## REPETITIVE PROCESS OF OBTAINING AND PROCESSING SENSOR DATA ##
for i in range(1000):
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
        print("Error: No Image Frame Obtained From The Webcam")
        break

    ## GET LOCALIZATIONS ACCORDING TO ERRORS, COMBINE WITH TIMESTAMPS ##
    wc_xyz = localize_aruco(wc_frame)
    if wc_xyz is None:
        print("skipping Webcam frame: no translation vector could be obtained in this frame.")
        pass
    else:
        wc_xyzt = np.append(wc_xyz, wc_t)
        print(wc_xyzt)
        
        ## PLOT THE LOCALIZATIONS ## 
        #(only 2D for now)
        sc.set_offsets(-1 * wc_xyzt[[0,1]]) #only select what is "links-rechts" and "oben-unten" axes and mirror it
        fig.canvas.draw_idle()
        plt.pause(0.01)
    
## TURN OFF / DISCONNECT ALL SENSORS ##    
wc_cap.release()


#--------------------------------



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


#----------
# Mein 3D Scatterplot Versuche
# Bei einem ähnlichen code kriege ich wie hier erstmal nur graues Fenster
# Also vielleicht liegt es an Spyder oder macOS oder so?
plt.ion()

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Links_Rechts')
ax.set_ylabel('Entfernung')
ax.set_zlabel('Unten_Oben')

axes = plt.gca()
axes.set_xlim([-0.4,0.4])
axes.set_ylim([0, 7])
axes.set_zlim([-0.4,0.4])

scat = ax.scatter(tvecs[0:4,0], tvecs[0:4,2], -1 * tvecs[0:4,1], marker='o')


plt.show()

for i in [4,5,6]:
    offsets = tuple(tvecs[i,:])
    print(offsets)
    scat._offsets3d = offsets
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    fig.canvas.draw_idle()
    plt.pause(1)

#----------
# Funktionierendes 2D-Scatterplot Beispiel:
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(0,10)
plt.ylim(0,10)

plt.draw()
for i in range(30):
    x.append(np.random.rand(1)*10)
    y.append(np.random.rand(1)*10)
    print(x)
    print(y)
    offsets = np.c_[x,y]
    print(offsets)
    sc.set_offsets(offsets[-1:])
    #sc.set_offsets(np.c_[x,y])
    fig.canvas.draw_idle()
    plt.pause(0.5)

plt.waitforbuttonpress()
#------------------------------
# 3D Scatterplot Updateting, was laut eines github users funktionieren sollte
# Aber genau wie bei meinem 3d plot bekomme ich nur ein graues Fenster
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
    planets._offsets3d(positions[:,0],positions[:,1],positions[:2])

ani=FuncAnimation(fig,animate,frames=1000,interval=1,blit=False)

#------------------------


x = np.linspace(0, 6*np.pi, 1000)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
    plt.pause(0.01)
   # fig.canvas.flush_events()
    
    
    
    
    
    
X = np.linspace(0,2,1000)
Y = X**2 + np.random.random(X.shape)

plt.ion()
graph = plt.plot(X,Y)[0]

while True:
    Y = X**2 + np.random.random(X.shape)
    graph.set_ydata(Y)
    plt.draw()
    plt.pause(0.01)




import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):
    print(phase)
    line1.set_ydata(np.sin(0.5 * x + phase))
    #plt.display(fig)
    fig.canvas.draw()
    fig.canvas.flush_events()
    

########


plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(0,10)
plt.ylim(0,10)

plt.draw()
for i in range(1000):
    x.append(np.random.rand(1)*10)
    y.append(np.random.rand(1)*10)
    print(x)
    print(y)
    offsets = np.c_[x,y]
    print(offsets)
    sc.set_offsets(offsets[-1:])
    #sc.set_offsets(np.c_[x,y])
    fig.canvas.draw_idle()
    plt.pause(1)

plt.waitforbuttonpress()

#######

from matplotlib.animation import FuncAnimation
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





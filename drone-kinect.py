#!/usr/bin/env python3

from libfreenect2 import kinect
from libfreenect2.modules import aruco, depth

from filters import kalman

import cv2
import numpy as np
import sys


if('--no-aruco' not in sys.argv):
	kinect.register(aruco, '--render-aruco' in sys.argv)
if('--no-depth' not in sys.argv):
	kinect.register(depth, '--render-depth' in sys.argv)

while True:
	kmr = kinect.getData()
	#print('dtime %f : %s' % (kmr.dtime.total_seconds(), str(kmr.coords)))
	filtered = kalman.filter(kmr.dtime, kmr.coords)
	print('3D coords: %s' % (filtered))

	# frame progression for rendered modules
	cv2.waitKey(1)


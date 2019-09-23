#!/usr/bin/env python3

import argparse
import cv2
import numpy as np

from filters import kalman

parser = argparse.ArgumentParser(description='Track a drone via Kinect v2 camera')
# module disable flags
parser.add_argument('--no-kinect', dest='sensor_kinect_disable', default=False, action='store_true', help='Disable the kinect sensor and all its modules')
parser.add_argument('--no-aruco', dest='module_aruco_disable', default=False, action='store_true', help='Disable aruco module')
parser.add_argument('--no-depth', dest='module_depth_disable', default=False, action='store_true', help='Disable depth module')
# module render flags
parser.add_argument('--render-aruco', dest='module_aruco_render', default=False, action='store_true', help='Render aruco image estimate')
parser.add_argument('--render-depth', dest='module_depth_render', default=False, action='store_true', help='Render depth image estimate')

# depth args
parser.add_argument('--min-depth', dest='module_depth_min', nargs=1, type=int, default=500, metavar='N', help='Lower bound of the detection range in millimeters')
parser.add_argument('--max-depth', dest='module_depth_max', nargs=1, type=int, default=1000, metavar='N', help='Upper bound of the detection range in millimeters')

# webcam sensor
parser.add_argument('--use-webcam', dest='sensor_webcam_enable', default=False, action='store_true', help='Enable the webcam sensor')

args = parser.parse_args()

enabled_sensors = []

if not args.sensor_kinect_disable:
	from sensors import kinect
	enabled_sensors.append(kinect)
	if not args.module_aruco_disable:
		from libfreenect2.modules import aruco
		kinect.register(aruco, args.module_aruco_render)
	if not args.module_depth_disable:
		from libfreenect2.modules import depth
		depth.set_range(args.module_depth_min, args.module_depth_max)
		kinect.register(depth, args.module_depth_render)

if args.sensor_webcam_enable:
	# TODO implement webcam sensor, add aruco module
	# TODO make modules global, create data_source enum as module_required_input
	pass


while True:	
	# TODO sensors should run in separate threads and send data directly to filter
	datapoints = []
	for sensor in enabled_sensors:
		datapoints.append(sensor.getData())
	
	if len(datapoints) == 0:
		print('No data')
		continue

	if len(datapoints[0].coords) > 0:
		filtered = kalman.filter(datapoints[0].dtime, datapoints[0].coords)
		print('3D coords: %s' % (filtered))
	else:
		print('3D coords: NO DATA')

	# frame progression for rendered modules
	cv2.waitKey(1)


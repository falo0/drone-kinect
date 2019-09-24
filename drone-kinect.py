#!/usr/bin/env python3

from filters import kalman
from datatypes import Datatype, SensorData


import argparse
import cv2
import numpy as np
import time


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

# live plot
parser.add_argument('--live-plot', dest='live_plot_enable', default=False, action='store_true', help='Enable live plotting of 3D estimation')


args = parser.parse_args()


if not args.sensor_kinect_disable:
	from sensors import kinect
	if not args.module_aruco_disable:
		from modules import aruco
	if not args.module_depth_disable:
		from modules import depth
		depth.set_range(args.module_depth_min, args.module_depth_max)

if args.sensor_webcam_enable:
	from sensors import webcam
	from modules import aruco

if args.live_plot_enable:
	from tools import liveplot

while True:
	results = []
	if not args.sensor_kinect_disable:
		# kinect_data as defined in kinect.KinectData
		kinect_data = kinect.getData()
		kinect_time = time.time()
		if not args.module_aruco_disable:
			results.append(
				SensorData(
					Datatype.KINECT_ARUCO,
					aruco.calc(kinect_data, flip=True, render=args.module_aruco_render),
					kinect_time
				)
			)
		if not args.module_depth_disable:
			results.append(
				SensorData(
					Datatype.KINECT_DEPTH,
					depth.calc(kinect_data, args.module_depth_render),
					kinect_time
				)
			)

	if args.sensor_webcam_enable:
		# webcam data as defined in webcam.WebcamData
		webcam_data = webcam.getData()
		webcam_time = time.time()
		if not args.module_aruco_disable:
			results.append(
				SensorData(
					Datatype.WEBCAM_ARUCO,
					aruco.calc(webcam_data, flip=False, render=args.module_aruco_render),
					webcam_time
				)
			)

	filtered_results = kalman.filter(results, method='velocity')
	print('kalman result: %s' % ( str(filtered_results) ) )

	if args.live_plot_enable:
		plot_points = []
		for i in results:
			if i.data is None:
				continue
			if i.datatype == Datatype.KINECT_DEPTH:
				plot_points.append(np.array([0, 0, i.data[0]]))
			else:
				plot_points.append(i.data)

		plot_points.append(filtered_results)
		coords = np.vstack(plot_points)
		colors = ['r', 'b', 'g']
		liveplot.update_3dplot(np.vstack(plot_points), colors)


	# frame progression for rendered modules
	cv2.waitKey(1)


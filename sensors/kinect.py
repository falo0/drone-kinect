#!/usr/bin/env python3

from libfreenect2 import kinect_wrapper
import numpy as np

class KinectData:
	'''
	Single instance data object that is passed to processing modules.
	Members are overwritten in c++ in every getData().
	'''
	rgb = np.zeros((1080, 1920, 4), dtype=np.uint8)
	ir = np.zeros((424, 512, 4), dtype=np.uint8) #TODO change datatype here and in c++
	depth = np.zeros((424, 512), dtype=np.float32)
	registered = np.zeros((424, 512, 4), dtype=np.uint8)


_k = kinect_wrapper.Kinect()
_kd = KinectData()

def getData():

	_k.getData(_kd.rgb, _kd.ir, _kd.depth, _kd.registered)

	return _kd



#!/usr/bin/env python3

from libfreenect2 import kinect_wrapper
import numpy as np

from datetime import datetime

class KinectData:
	'''
	Single instance data object that is passed to processing modules.
	Members are overwritten in c++ in every getData().
	'''
	rgb = np.zeros((1080, 1920, 4), dtype=np.uint8)
	ir = np.zeros((424, 512, 4), dtype=np.uint8) #TODO change datatype here and in c++
	depth = np.zeros((424, 512), dtype=np.float32)
	registered = np.zeros((424, 512, 4), dtype=np.uint8)


class KinectModuleResult:
	'''
	Single instance result data object that is overwritten in getData() and returned to callee.
	'''
	dtime = None
	coords = None

_k = kinect_wrapper.Kinect()
_kd = KinectData()
_kmr = KinectModuleResult()

_registered_modules = []

def register(module, render=False):
	'''
	Register a processing module that has a calc(KinectData_obj, render_bool) method.
	Do NOT modify the object and return a [x,y,z] result estimate.
	All results are collected and returned to the getData() callee.
	'''
	_registered_modules.append({'module': module, 'render': render})


_last_timestamp = datetime.now()

def getData():
	'''
	Returns 2D array [[x1,y1,z1], ... , [xn,yn,zn]].
	Contains results of all registered modules
	'''

	_k.getData(_kd.rgb, _kd.ir, _kd.depth, _kd.registered)

	global _last_timestamp
	cur_timestamp = datetime.now()
	_kmr.dtime = cur_timestamp - _last_timestamp
	_last_timestamp = cur_timestamp

	_kmr.coords = []
	for mod in _registered_modules:
		module = mod['module']
		module_res = module.calc(_kd, mod['render'])
		if module_res is not None:
			_kmr.coords.append(module_res)
	
	return _kmr



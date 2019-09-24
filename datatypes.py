from enum import Enum


class Datatype(Enum):

	KINECT_ARUCO = 'kinect aruco'
	KINECT_DEPTH = 'kinect depth'

	WEBCAM_ARUCO = 'webcam aruco'


class SensorData():
	def __init__(self, datatype, data, timestamp):
		self.datatype = datatype
		self.data = data
		self.timestamp = timestamp



import cv2

webcam_capture = cv2.VideoCapture(0)

class WebcamData:
	rgb = None

_webcam_data = WebcamData()

def getData():
	read_flag, frame = webcam_capture.read()
	if frame == None:
		raise IOError('No data from webcam.')
	_webcam_data.rgb = frame
	return _webcam_data


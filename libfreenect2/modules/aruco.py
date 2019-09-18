import cv2

def calc(data, render):
	frame = data.rgb
	frame = cv2.flip(frame[:,:,0:3], 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

	if render:
		cv2.imshow(__name__, frame_markers)
	
	if len(corners) > 0:
		#TODO calc marker distance and center
		return corners[0]
	return None


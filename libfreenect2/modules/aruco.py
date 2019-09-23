import cv2


# side length of the used ArUco marker: 0.105 meters
marker_length = 0.105

cam_config_file = cv2.FileStorage("camera_calibration/MBP13Late2015Distortion.yaml", cv2.FILE_STORAGE_READ)
matrix_coefficients = cam_config_file.getNode("camera_matrix").mat()
distortion_coefficients = cam_config_file.getNode("dist_coeff").mat()

def calc(data, render):
	global matrix_coefficients
	global distortion_coefficients

	frame = cv2.flip(data.rgb[:,:,0:3], 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	param = cv2.aruco.DetectorParameters_create()
	dictio = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictio, parameters=param)
    
	global marker_length
	rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, matrix_coefficients, distortion_coefficients)
	if tvec is not None:
		return tvec[0,0] #assuming there is only one marker in the image
	return None

	# TODO re-add rendering

	if render:
		cv2.imshow(__name__, frame_markers)
	
	if len(corners) > 0:
		#TODO calc marker distance and center
		return corners[0]
	return None


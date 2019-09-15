import cv2

def calc(data, render):
	if render:
		cv2.imshow(__name__, data.depth)
	return [1, 1, 1]


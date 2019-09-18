import cv2
import numpy as np

def calc(data, render):
	
	# TODO set depth via c++ wrapper and calc final depth based on (0, 1] multiplier of distance diff

	a = np.where(data.depth > 0)
	if(len(a[0]) == 0):
		return None

	x_mean = int(a[0].mean())
	y_mean = int(a[1].mean())

	if render:
		i_rgb = cv2.cvtColor(data.depth, cv2.COLOR_GRAY2RGB)
		cv2.circle(i_rgb, (y_mean, x_mean), 10, (255, 0 , 0), 5)
		cv2.imshow(__name__, i_rgb)
	return [x_mean, y_mean, data.depth[x_mean, y_mean]]


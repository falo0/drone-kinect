#!/usr/bin/env python3

import pyfreenect2
import cv2
import numpy as np

pf = pyfreenect2.PyFreeNect2()

cv2.startWindowThread()
cv2.namedWindow("RGB")
while True:
    data = pf.get_new_frame(get_BGR = True)
    cv2.imshow('RGB', frame.GBR)
    cv2.waitKey(1)
print("done")


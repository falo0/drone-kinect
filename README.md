# Drone Kinect Tracking

## Project Description:
The goal of the project is to localize a drone. Various sensors might be used, like RGB-camera (webcam or RGB-cam of Kinect V2), depth sensor (Kinect V2), IMU of the drone, IR camera (of Kinect V2).

## To Do:
- 'ArUco localization' of an ArUco marker sticked to the drone should work first. We want (x,y,z)-coordinates (of the center of the ArUco marker). It would be great if it worked with Kinect and normal webcams, so at least parts of the project can be used when no Kinect V2 is available.

- Also do ArUco localization with the IR camera if the projected IR points are not distracting too much. This way there is a localization based on the images of 2 different cameras.

- Getting another source of the z- or depth-coordinate from the depth sensor: From OpenCV/ArUco we should know the pixel (x,y)-coordinates/indices of the center of the ArUco marker. We can look up the depth/z value at the same spot/pixel of the depth image.
-- Getting also x and y coordinates from the depth values alone would require drone tracking based on depth alone, so no ArUco tracking would be possible

- We now have (x,y,z)-coordinates derived from 2 cameras and also z-coordinates from the depth sensor. Can we fuse these data to better estimate the true (x,y,z)-coordinate? We might use a Kalman filter, also considering past location estimates.

If time is left:
- Task A) Trying to get rid of the ArUco marker. If we only have a white background and a black drone it might be possible to detect the drone by brightness differences. If we have any steady background where the drone is the only thing moving, we can also detect the drone by comparing to the previous image and identifying the pixels that changed. If we have more challenging backgrounds, it's much harder but it can still be tried to approach it. We will need to train a neural net what a drone is at all orientations at which the drone might be in front of the camera. We need to draw the drone's contour on a lot of drone pictures by hand to get the training data.

- Task B) If there is access to the IMU sensor data from the drone, we might also do IMU localization (this is data fusion from accelerometer, gyroscope, magnetometer, barometer, ...), just so we have a third location estimate to fuse with the camera and depth sensor data.

- Task C) Make use of the microphone array of the Kinect. We could try sound positioning to at least get another source of (x,y)-coordinates, maybe even z-coordinate based on the amplitude of the drone noise.

## Requirements

python-numpy
libfreenect2
libfreenect2 python wrapper from https://github.com/37/py3freenect2/
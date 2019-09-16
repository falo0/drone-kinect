#libfreenect2 Interface

A wrapper that handles data retrieval and processing from the libfreenect2 library.

Built as a shared library and imported by `kinect.py`.

Data Processing has a modular structure. Register your data processing module with `kinect.register` and implement a method calc(kinect.KinectData, render\_boolean) that calculates an estimated 3D coordinate and returns it as a [x, y, z] array.


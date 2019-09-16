#!/bin/sh

cd $(dirname $0)

g++ kinect_wrapper.cpp $(python3-config --includes) -fpic $(python3-config --ldflags) -lboost_python3 -lboost_numpy3 -lfreenect2 -shared -o kinect_wrapper.so


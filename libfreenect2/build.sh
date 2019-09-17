#!/bin/sh

cd $(dirname $0)

g++ kinect_wrapper.cpp $(python3-config --includes) -fpic -L/Users/Dolan/anaconda3/lib/python3.6/config-3.6m-darwin -lpython3.6m -ldl -framework CoreFoundation -Wl -framework CoreFoundation -lboost_python3 -lboost_numpy3 -lfreenect2 -shared -o kinect_wrapper.so


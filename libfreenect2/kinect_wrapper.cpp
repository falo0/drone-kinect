
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/frame_listener_impl.h>

#include <iostream>
#include <signal.h>

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
//#include <algorithm> //std::min max

namespace p = boost::python;
namespace np = boost::python::numpy;

class Kinect {
public:
	libfreenect2::Freenect2 *freenect2;
	libfreenect2::Freenect2Device *dev = 0;
	libfreenect2::PacketPipeline *pipeline = 0;
	libfreenect2::Registration* registration = nullptr;

	libfreenect2::SyncMultiFrameListener *listener;
	libfreenect2::FrameMap frames;

	libfreenect2::Frame *undistorted = nullptr, *registered = nullptr;

	Kinect(){

		int deviceId = -1;
		freenect2 = new libfreenect2::Freenect2();

		if(freenect2->enumerateDevices() == 0) {
			std::cout << "Kinect not connected!" << std::endl;
			return;
		}

		std::string serial = freenect2->getDefaultDeviceSerialNumber();
		//pipeline = new libfreenect2::DumpPacketPipeline();
		pipeline = new libfreenect2::OpenGLPacketPipeline();
		dev = freenect2->openDevice(serial, pipeline);

		if(dev == 0) {
			std::cout << "Failed to open device!" << std::endl;
			return;
		}

		int types = 0;
		types |= libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
		this->listener = new libfreenect2::SyncMultiFrameListener(types);

		dev->setColorFrameListener(listener);
		dev->setIrAndDepthFrameListener(listener);

		if (!dev->start())
			return;

		std::cout << "Device serial: " << dev->getSerialNumber() << std::endl;
		std::cout << "Device firmware: " << dev->getFirmwareVersion() << std::endl;

		registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
		undistorted = new libfreenect2::Frame(512, 424, 4);
		registered = new libfreenect2::Frame(512, 424, 4);

	}

	~Kinect(){
		dev->stop();
		dev->close();
		delete this->registration;
		delete this->undistorted;
		delete this->registered;
	}

	void getData(np::ndarray &input_rgb_np, np::ndarray &input_ir_np, np::ndarray &input_depth_np, np::ndarray &input_registered_np) {
		if (!listener->waitForNewFrame(this->frames, 10*1000)) {
			std::cout << "timeout!" << std::endl;
		}
		libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
		libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
		libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

		registration->apply(rgb, depth, undistorted, registered);

		// fill input array
		unsigned char* input_rgb = reinterpret_cast<unsigned char*>(input_rgb_np.get_data());
		unsigned char* input_ir = reinterpret_cast<unsigned char*>(input_ir_np.get_data());
		//unsigned char* input_depth = reinterpret_cast<unsigned char*>(input_depth_np.get_data());
		float* input_depth = reinterpret_cast<float*>(input_depth_np.get_data());
		unsigned char* input_registered = reinterpret_cast<unsigned char*>(input_registered_np.get_data());

		memcpy(input_rgb, rgb->data , 1920*1080*4*sizeof(unsigned char));
		memcpy(input_ir, ir->data , 512*424*4*sizeof(unsigned char));
		memcpy(input_depth, depth->data , 512*424* 4);

		//min,max distance in mm
		float depth_min = 500;
		float depth_max = 1000;
		for(int i=0;i<512*424; i++){
			input_depth[i] = (std::max(depth_min, std::min(input_depth[i], depth_max)) - depth_min) / (depth_max-depth_min);
			if(input_depth[i] == 1.0f){
				input_depth[i] = 0;
			}
		}

		//memcpy(input_registered, registered->data , 512*424*4*sizeof(unsigned char));
		memcpy(input_registered, undistorted->data , 512*424*4*sizeof(unsigned char));

		listener->release(frames);
	}

	};


	BOOST_PYTHON_MODULE(kinect_wrapper){
		using namespace boost::python;
		np::initialize();
		class_<Kinect>("Kinect")
			.def("getData", &Kinect::getData);
		//	.def("destroy", &Kinect::destroy);

	}




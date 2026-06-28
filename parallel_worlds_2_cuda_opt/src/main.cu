#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdlib>
#include <iostream>

#include "CUDAInterface.hpp"
#include "EffectFilter.hpp"
#include "GreyFilter.hpp"
#include "SobelFilter.hpp"

__host__ int main(int argc, const char** argv) {
	GreyFilter greyFilter(3,1);
	SobelFilter sobelFilter(1, 1);
	EffectFilter effectFilter(3, 1, 3);

	cv::VideoCapture capture(0); //0=default, -1=any camera, 1..99=your camera
	cv::Mat frame;

	bool cameraOn = capture.isOpened() && false;
	if (cameraOn) {
		if (!capture.read(frame))
			exit(3);
	} else {
		std::cerr << "No camera detected" << std::endl;
		frame = cv::imread("preview.png");
		if(frame.data == NULL)
			exit(3);
	}

	const unsigned int w = frame.cols;
	const unsigned int h = frame.rows;

	cv::Mat convertedFrame(h, w, CV_8UC1);
	cv::Mat edgeFrame(h, w, CV_8UC1);
	cv::Mat effectFrame(h, w, CV_8UC3);

	const size_t colorBytes = static_cast<size_t>(w) * h * frame.elemSize();
	const size_t greyBytes = static_cast<size_t>(w) * h * convertedFrame.elemSize();
	unsigned char *dFrame = nullptr;
	unsigned char *dGreyFrame = nullptr;
	unsigned char *dEdgeFrame = nullptr;
	unsigned char *dEffectFrame = nullptr;
	const bool hasDisplay = std::getenv("DISPLAY") != nullptr
			|| std::getenv("WAYLAND_DISPLAY") != nullptr;

	SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dFrame), colorBytes));
	SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dGreyFrame), greyBytes));
	SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dEdgeFrame), greyBytes));
	SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dEffectFrame), colorBytes));

	if (hasDisplay) {
		cv::namedWindow("preview", 0);
		cv::namedWindow("effect", 0);
	}

	if (!hasDisplay) {
		SAFE_CALL(cudaMemcpy(dFrame, frame.data, colorBytes, cudaMemcpyHostToDevice));
		greyFilter.applyDevice(dFrame, dGreyFrame, w, h);
		sobelFilter.applyDevice(dGreyFrame, dEdgeFrame, w, h, .5f);
		effectFilter.applyDevice(dFrame, dEdgeFrame, dEffectFrame, w, h, 90.0f);
		SAFE_CALL(cudaMemcpy(effectFrame.data, dEffectFrame, colorBytes, cudaMemcpyDeviceToHost));
		SAFE_CALL(cudaDeviceSynchronize());

		if (!cv::imwrite("effect_output.png", effectFrame)) {
			std::cerr << "Could not write effect_output.png" << std::endl;
			return 4;
		}
		std::cout << "No display detected. Wrote effect_output.png" << std::endl;
	} else {
		while (((char)cv::waitKey(10)) <= -1) {
			if (cameraOn && !capture.read(frame))
				exit(3);

			SAFE_CALL(cudaMemcpy(dFrame, frame.data, colorBytes, cudaMemcpyHostToDevice));
			greyFilter.applyDevice(dFrame, dGreyFrame, w, h);
			sobelFilter.applyDevice(dGreyFrame, dEdgeFrame, w, h, .5f);
			effectFilter.applyDevice(dFrame, dEdgeFrame, dEffectFrame, w, h, 90.0f);
			SAFE_CALL(cudaMemcpy(effectFrame.data, dEffectFrame, colorBytes, cudaMemcpyDeviceToHost));
			SAFE_CALL(cudaDeviceSynchronize());

			cv::imshow("preview", frame);
			cv::imshow("effect", effectFrame);
		}
	}

	if (hasDisplay) {
		cv::destroyWindow("preview");
		cv::destroyWindow("effect");
	}

	SAFE_CALL(cudaFree(dFrame));
	SAFE_CALL(cudaFree(dGreyFrame));
	SAFE_CALL(cudaFree(dEdgeFrame));
	SAFE_CALL(cudaFree(dEffectFrame));

	return 0;
}

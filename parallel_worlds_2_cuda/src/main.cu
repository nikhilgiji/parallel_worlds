#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "CUDAInterface.hpp"
#include "GreyFilter.hpp"
#include "SobelFilter.hpp"

__host__ int main(int argc, const char** argv) {
	GreyFilter greyFilter(3,1);
	SobelFilter sobelFilter(1, 1);

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

	// resulting image after conversion is greyscale
	cv::Mat convertedFrame(h, w, CV_8UC1);

	// resulting image after sobelColor is color
	cv::Mat edgeFrame(h, w, CV_8UC1);
	const bool hasDisplay = std::getenv("DISPLAY") != nullptr
			|| std::getenv("WAYLAND_DISPLAY") != nullptr;
	std::size_t frameCount = 0;
	double totalMs = 0.0;

	if (hasDisplay) {
		cv::namedWindow("preview", 0);
		cv::namedWindow("converted", 0);
	}

	auto runPipeline = [&]() {
		const auto start = std::chrono::steady_clock::now();
		greyFilter(frame.data, convertedFrame.data, w, h);
		sobelFilter(convertedFrame.data, edgeFrame.data, w, h, .5f);
		const auto end = std::chrono::steady_clock::now();
		const double frameMs = std::chrono::duration<double, std::milli>(end - start).count();
		totalMs += frameMs;
		++frameCount;
		if (frameCount % 30 == 0) {
			std::cout << "Average pipeline time over " << frameCount
					<< " frames: " << (totalMs / frameCount) << " ms" << std::endl;
		}
		return frameMs;
	};

	if (!hasDisplay) {
		const double frameMs = runPipeline();
		if (!cv::imwrite("converted_output.png", edgeFrame)) {
			std::cerr << "Could not write converted_output.png" << std::endl;
			return 4;
		}
		std::cout << "No display detected. Wrote converted_output.png" << std::endl;
		std::cout << "Pipeline time: " << frameMs << " ms" << std::endl;
	} else {
		while (((char)cv::waitKey(10)) <= -1) {
			if (cameraOn && !capture.read(frame))
				exit(3);

			runPipeline();
			// show the result
			cv::imshow("preview", frame);
			cv::imshow("converted", edgeFrame);
		}
	}

	if (hasDisplay) {
		cv::destroyWindow("preview");
		cv::destroyWindow("converted");
	}

	return 0;
}

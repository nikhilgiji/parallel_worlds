#include "EffectFilter.hpp"

#define ARR(A,x,y,maxX) (A[(x)+(y)*(maxX)])
#define ARRC(A,x,y,maxX,channel) (A[((x)+(y)*(maxX))*3+channel])

__global__ void effectKernel(const unsigned char *inImg, const unsigned char *edgeImg,
		unsigned char *outImg, unsigned int w, unsigned int h, float threshold) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		const signed char G = static_cast<signed char>(ARR(edgeImg, x, y, w));
		const float absG = fabsf(static_cast<float>(G));
		ARRC(outImg, x, y, w, 0) = ARRC(inImg, x, y, w, 0);
		ARRC(outImg, x, y, w, 1) = ARRC(inImg, x, y, w, 1);
		ARRC(outImg, x, y, w, 2) = ARRC(inImg, x, y, w, 2);
		if (absG > threshold) {
			ARRC(outImg, x, y, w, 0) =
					static_cast<unsigned char>(ARRC(outImg, x, y, w, 0) * 0.5f);
			ARRC(outImg, x, y, w, 1) =
					static_cast<unsigned char>(ARRC(outImg, x, y, w, 1) * 0.5f);
			ARRC(outImg, x, y, w, 2) =
					static_cast<unsigned char>(ARRC(outImg, x, y, w, 2) * 0.5f);
		}
	}
}

__host__ void EffectFilter::applyDevice(const unsigned char *deviceInput,
		const unsigned char *deviceEdgeInput, unsigned char *deviceOutput,
		unsigned int currWidth, unsigned int currHeight, float threshold) {
	this->prepareLaunch(currWidth, currHeight);
	effectKernel<<<this->grid, this->threads>>>(deviceInput, deviceEdgeInput,
			deviceOutput, currWidth, currHeight, threshold);
}

__host__ void EffectFilter::operator()(const unsigned char *input,
		const unsigned char *edgeInput, unsigned char *output,
		unsigned int currWidth, unsigned int currHeight, float threshold) {
	this->prepareBuffers(currWidth, currHeight);
	SAFE_CALL(cudaMemcpy(this->dInput, reinterpret_cast<const void*>(input),
			currWidth * currHeight * this->depthIn, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(this->dEdgeInput, reinterpret_cast<const void*>(edgeInput),
			currWidth * currHeight * this->depthEdge, cudaMemcpyHostToDevice));
	this->applyDevice(this->dInput, this->dEdgeInput, this->dOutput,
			currWidth, currHeight, threshold);
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), this->dOutput,
			currWidth * currHeight * this->depthOut, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());
}

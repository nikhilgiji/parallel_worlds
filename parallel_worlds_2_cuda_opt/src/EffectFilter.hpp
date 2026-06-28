#ifndef EFFECTFILTER_HPP_
#define EFFECTFILTER_HPP_

#include "ImageFilter.hpp"

/// Implements an effect based on edge detection.
class EffectFilter : public ImageFilter {
protected:
	unsigned char *dEdgeInput;
	unsigned int depthEdge;
public:
	EffectFilter(unsigned int dIn, unsigned int dEdge, unsigned int dOut) :
		ImageFilter(dIn, dOut), dEdgeInput(nullptr), depthEdge(dEdge) {
	}

	virtual ~EffectFilter() {
		SAFE_CALL(cudaFree(dEdgeInput));
	}

	virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) override {
		const unsigned int bytesEdge = currWidth * currHeight * depthEdge;
		this->prepareLaunch(currWidth, currHeight);
		if (currWidth * currHeight > width * height) {
			SAFE_CALL(cudaFree(dEdgeInput));
			SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dEdgeInput), bytesEdge));
		}
		ImageFilter::resizeBuffers(currWidth, currHeight);
	}

	void applyDevice(const unsigned char *deviceInput, const unsigned char *deviceEdgeInput,
			unsigned char *deviceOutput, unsigned int currWidth,
			unsigned int currHeight, float threshold);
	void operator()(const unsigned char *input, const unsigned char *edgeInput,
			unsigned char *output, unsigned int currWidth,
			unsigned int currHeight, float threshold);
};

#endif /* EFFECTFILTER_HPP_ */

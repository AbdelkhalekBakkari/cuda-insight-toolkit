float * CudaGrayscaleErodeImageFilterKernelFunction(const float* input,
				const unsigned long * imageDim, const unsigned long * radius,
				const float * kernel, const unsigned long * kernelDim, const float zero,
				unsigned long D, unsigned long N);


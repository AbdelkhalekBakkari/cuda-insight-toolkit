template <class T, class S> extern
void CudaRescaleIntensityKernelFunction(const float* input, S* output,
					S outputMax, 
					S outputMin,
					unsigned int N);


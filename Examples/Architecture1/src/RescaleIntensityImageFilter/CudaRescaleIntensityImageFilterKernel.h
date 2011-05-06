template <class T, class S> extern
void CudaRescaleIntensityKernelFunction(const T* input, S* output,
				    S outputMax, 
				    S outputMin,
				    unsigned int N);


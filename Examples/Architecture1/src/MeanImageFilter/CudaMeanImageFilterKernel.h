template <class T, class S> extern
void CudaMeanImageFilterKernelFunction(const T* input, S *output, unsigned int imageDimX, 
				       unsigned int imageDimY, unsigned int imageDimZ,
				       unsigned int radiusX, unsigned int radiusY, 
				       unsigned int radiusZ, unsigned int N);

template <class T, class S, class K> extern
void CudaGrayscaleDilateImageFilterKernelFunction(const T* input, S* output,
						  const unsigned long * imageDim, 
						  const unsigned long * radius,
						  const K * kernel, 
						  const unsigned long * kernelDim, const K zero,
						  unsigned long D, unsigned long N);

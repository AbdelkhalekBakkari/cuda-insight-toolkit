/*
 * File Name:    cuda-kernel.h
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, January 18 2010, 10:00 
 * Last Modified: Thursday, January 14 2010, 15:58
 * 
 * File Description:
 *
 */
float * CudaGrayscaleDilateImageFilterKernelFunction(const float* input,
				const unsigned long * imageDim, const unsigned long * radius,
				const float * kernel, const unsigned long * kernelDim, const float zero,
				unsigned long D, unsigned long N);


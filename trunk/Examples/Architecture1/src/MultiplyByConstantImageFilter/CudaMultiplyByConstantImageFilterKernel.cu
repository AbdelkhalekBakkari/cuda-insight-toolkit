/*
 * File Name:    cuda-kernel.cu
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, January 18 2010, 10:00 
 * Last Modified: Wednesday, December 23 2009, 16:35 
 * 
 * File Description:
 *
 */
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

#include "CudaMultiplyByConstantImageFilterKernel.h"

template <class T>
__global__ void MultiplyByConstantImageKernel(T *output, int N, T C) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		output[idx] *= C;
	}
}

CITK_OUT * MultiplyByConstantImageKernelFunction(const CIT_IN1* input1,
		unsigned int N, CITK_IN1 C) {
	CITK_OUT *output;

	output = const_cast<CITK_IN1*> (input1);

	// Compute execution configuration
	int blockSize = 128;
	int nBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	// Call kernel
	MultiplyByConstantImageKernel <<< nBlocks, blockSize >>> (output, N, C);

	// Return pointer to the output
	return output;
}

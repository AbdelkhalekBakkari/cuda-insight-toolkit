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

template <class T, class S>
__global__ void DivideImageKernel(T *output, const S *input, int N, T MAX)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = output[idx] / input[idx];
      if (input[idx]==0) { output[idx] = MAX; }
   }
}

float * DivideImageKernelFunction(const float* input1, const float* input2, unsigned int N, float MAX)
{
   float *output;

   output = const_cast<float*>(input1);

   // Compute execution configuration 
   int blockSize = 64;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernal
   DivideImageKernel <<< nBlocks, blockSize >>> (output, input2, N, MAX);

   // Return pointer to the output
   return output;
}

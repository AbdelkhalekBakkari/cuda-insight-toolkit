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
__global__ void MaximumImageKernel(T *output, const S *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
	   S inputValue = input[idx];
      if (output[idx] < inputValue)
      {
         output[idx] = inputValue;
      }
   }
}

float * MaximumImageKernelFunction(const float* input1, const float* input2, unsigned int N)
{
   float *output;

   output = const_cast<float*>(input1);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernal
   MaximumImageKernel <<< nBlocks, blockSize >>> (output, input2, N);

   // Return pointer to the output
   return output;
}

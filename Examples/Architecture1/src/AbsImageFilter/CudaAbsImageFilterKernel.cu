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
//#include <cutil.h>

template <class T>
__global__ void AbsImageKernel(T*output, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx<N) 
   {
	   T temp = output[idx];
      output[idx] = (temp < 0) ? -temp : temp;
   }
}

float * AbsImageKernelFunction(const float * input, unsigned int N)
{
	float * output = const_cast<float*>(input);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernal
   AbsImageKernel<<< nBlocks, blockSize >>> (output, N);

   // Return pointer to the output
   return output;
}

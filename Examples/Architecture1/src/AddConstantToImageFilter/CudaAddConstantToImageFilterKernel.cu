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

template <class T>
__global__ void AddConstantToImageKernel(T *output, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] += C;
   }
}

float * AddConstantToImageKernelFunction(const float* input, unsigned int N, float C)
{
   float *output;

   output = const_cast<float*>(input);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernal
   AddConstantToImageKernel <<< nBlocks, blockSize >>> (output, N, C);

   // Return pointer to the output
   return output;
}

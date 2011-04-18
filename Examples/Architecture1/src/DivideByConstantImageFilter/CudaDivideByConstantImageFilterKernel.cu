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
__global__ void DivideByConstantImageKernel(T *output, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = output[idx] / C;
      
   }
}

float * DivideByConstantImageKernelFunction(const float* input1, unsigned int N, float C)
{
   float *output;

   output = const_cast<float*>(input1);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernal
   DivideByConstantImageKernel <<< nBlocks, blockSize >>> (output, N, C);

   // Return pointer to the output
   return output;
}

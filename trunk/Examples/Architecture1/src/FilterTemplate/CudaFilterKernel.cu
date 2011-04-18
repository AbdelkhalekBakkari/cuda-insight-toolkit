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
__global__ void cuKernel(T *output, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      // Do Something Here
   }
}

float * cuFunction(const float* input, unsigned int N)
{
   float *output;

   // Method #1 - Re-use Device Memory for Output
   //
   // Cast input to non const.
   // Note: ContainerManageDevice must be set to false in input container.
   // eg: 
   /*
      output = const_cast<float*>(input);
   */

   // Method #2 - Allocate New Memory for Output
   //
   // CudaMalloc new output memory
   // Note: ContainerManageDevice must be set to true in input container.
   // 
   // eg: 
   /* 
      cudaMalloc((void **) &output, sizeof(float)*N);
   */


   // Compute execution configuration 
   int blockSize = 64;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernal
   cuKernel <<< nBlocks, blockSize >>> (output, N);

   // Return pointer to the output
   return output;
}

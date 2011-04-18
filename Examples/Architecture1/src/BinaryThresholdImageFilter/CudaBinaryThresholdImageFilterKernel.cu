/*
 * File Name:    cuda-kernel.cu
 *
 * Author:        Phillip Ward
 * Creation Date: Wednesday, December 23 2009, 16:35 
 * Last Modified: Wednesday, December 23 2009, 16:35 
 * 
 * File Description:
 *
 */
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T>
__global__ void binaryThreshold(T *output, T lower, T upper, T inside, T outside, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      float i = output[idx];
      if (i>upper || i<lower)
      {
         output[idx] = inside;
      }
      else
      {
         output[idx] = outside;
      }
   }
}

float * BinaryThreshold(const float* input, float m_LowerThreshold, float m_UpperThreshold, float m_InsideValue, float m_OutsideValue, unsigned int N)
{
   // pointers
   //float *output;
   float *output = const_cast<float*>(input);

   // Compute execution configuration
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call  kernel
   binaryThreshold <<< nBlocks, blockSize >>> (output, m_LowerThreshold, m_UpperThreshold, m_InsideValue, m_OutsideValue, N);

   return output;
}

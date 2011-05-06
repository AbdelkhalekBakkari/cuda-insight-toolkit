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
__global__ void AddImageKernel(T *output, const S *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] += input[idx];
   }
}

template <class T, class S>
__global__ void AddImageKernel(T *output, const S *input1, const S* input2, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input1[idx] + input2[idx];
   }
}

template <class T, class S>
void AddImageKernelFunction(const T* input1, const T* input2, S* output, unsigned int N)
{


   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernal
   if (output == input1)
     AddImageKernel <<< nBlocks, blockSize >>> (output, input2, N);
   else
     AddImageKernel <<< nBlocks, blockSize >>> (output, input1, input2, N);
}

// versions we wish to compile
#define THISTYPE float
template void AddImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void AddImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void AddImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template void AddImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2,  THISTYPE *output, unsigned int N);
#undef THISTYPE

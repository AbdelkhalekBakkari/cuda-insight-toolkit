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

template <class S>
__global__ void SubtractConstantFromImageKernel(S *output, int N, S C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] -= C;
   }
}

template <class T, class S>
__global__ void SubtractConstantFromImageKernel(S *output, const T *input, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input[idx] - C;
   }
}

template<class T, class S>
void SubtractConstantFromImageKernelFunction(const T* input, S* output, unsigned int N, T C)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernal
   if (output == input)
     SubtractConstantFromImageKernel <<< nBlocks, blockSize >>> (output, N, C);
   else
     SubtractConstantFromImageKernel <<< nBlocks, blockSize >>> (output, input, N, C);

}

// versions we wish to compile
#define THISTYPE float
template void SubtractConstantFromImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, unsigned int N, THISTYPE C);
#undef THISTYPE
#define THISTYPE int
template void SubtractConstantFromImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE short
template void SubtractConstantFromImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE char
template void SubtractConstantFromImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE


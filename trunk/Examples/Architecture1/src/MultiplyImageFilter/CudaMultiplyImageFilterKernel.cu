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
__global__ void multiplyImage(T *output, const S *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] *= input[idx];
   }
}

template <class T, class S>
S * MultiplyImageKernelFunction(const T* input1, const T* input2, unsigned int N)
{
   S *output;

   output = const_cast<S*>(input1);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   multiplyImage <<< nBlocks, blockSize >>> (output, input2, N);

   // Return pointer to the output
   return output;
}

// versions we wish to compile
#define THISTYPE float
template THISTYPE *  MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template THISTYPE *  MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template THISTYPE *  MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template THISTYPE *  MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, unsigned int N);
#undef THISTYPE

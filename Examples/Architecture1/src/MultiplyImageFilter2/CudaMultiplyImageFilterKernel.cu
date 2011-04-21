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
__global__ void multiplyImage2(T *output, const S *input1, const S *input2, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input1[idx] * input2[idx];
   }
}


template <class T, class S>
void MultiplyImageKernelFunction(const T* input1, const T* input2, S *output, unsigned int N)
{
//   S *output;

//   output = const_cast<S*>(input1);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (output == input1)
     multiplyImage <<< nBlocks, blockSize >>> (output, input2, N);
   else
     multiplyImage2 <<< nBlocks, blockSize >>> (output, input1, input2, N);
     
   // Return pointer to the output
   //return output;
}


template void MultiplyImageKernelFunction<float, float>(const float *input1, const float *input2,  float *output, unsigned int N);

#if 0
// versions we wish to compile
#define THISTYPE float
template void MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, const THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, const THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, const THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template void MultiplyImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, const THISTYPE *output, unsigned int N);
#undef THISTYPE
#endif

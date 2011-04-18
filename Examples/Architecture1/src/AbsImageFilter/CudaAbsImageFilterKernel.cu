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
__global__ void AbsImageKernel(T *output, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx<N) 
   {
	   T temp = output[idx];
      output[idx] = (temp < 0) ? -temp : temp;
   }
}

template <class T, class S>
S* AbsImageKernelFunction(const T * input, unsigned int N)
{
	S * output;
	output = const_cast<S*>(input);

   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernal
   AbsImageKernel<<< nBlocks, blockSize >>> (output, N);

   // Return pointer to the output
   return output;
}

// versions we wish to compile
#define THISFUNC AbsImageKernelFunction
#define THISTYPE float
template THISTYPE *  AbsImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input,  unsigned int N);
#undef THISTYPE
#define THISTYPE int
template THISTYPE *  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template THISTYPE *  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template THISTYPE *  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input, unsigned int N);
#undef THISTYPE
#undef THISFUNC

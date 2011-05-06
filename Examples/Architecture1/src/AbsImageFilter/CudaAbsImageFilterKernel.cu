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
__global__ void AbsImageKernel(S *output, const T *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx<N) 
   {
   T temp = input[idx];
   output[idx] = (temp < 0) ? -temp : temp;
   }
}

template <class T, class S>
void AbsImageKernelFunction(const T * input, S * output, unsigned int N)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (input == output) 
     {
     AbsImageKernel<<< nBlocks, blockSize >>> (output, N);
     }
   else
     {
     AbsImageKernel<<< nBlocks, blockSize >>> (output, input, N);
     }

}

// versions we wish to compile
#define THISFUNC AbsImageKernelFunction
#define THISTYPE float
template void THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,   THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,  THISTYPE * output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,  THISTYPE * output, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template void  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,  THISTYPE * output, unsigned int N);
#undef THISTYPE
#undef THISFUNC

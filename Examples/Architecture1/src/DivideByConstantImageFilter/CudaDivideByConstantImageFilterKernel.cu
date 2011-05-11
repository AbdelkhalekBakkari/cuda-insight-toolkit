#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class S>
__global__ void DivideByConstantImageKernel(S *output, int N, S C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] /= C;
   }
}

template <class T, class S>
__global__ void DivideByConstantImageKernel(S *output, const T *input, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input[idx] / C;
   }
}

template<class T, class S>
void DivideByConstantImageKernelFunction(const T* input, S* output, unsigned int N, T C)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernel
   if (output == input)
     DivideByConstantImageKernel <<< nBlocks, blockSize >>> (output, N, C);
   else
     DivideByConstantImageKernel <<< nBlocks, blockSize >>> (output, input, N, C);

}
// versions we wish to compile
#define THISTYPE float
template void DivideByConstantImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE
 * output, unsigned int N, THISTYPE C);
#undef THISTYPE
#define THISTYPE int
template void DivideByConstantImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE
 *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE short
template void DivideByConstantImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE
 *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE unsigned char
template void DivideByConstantImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE
 *output, unsigned int N, THISTYPE C);
#undef THISTYPE

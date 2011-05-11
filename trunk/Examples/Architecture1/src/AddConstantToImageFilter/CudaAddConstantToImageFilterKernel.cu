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

#include <thrust/transform.h>

template <class S>
__global__ void AddConstantToImageKernel(S *output, int N, S C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] += C;
   }
}

template <class T, class S>
__global__ void AddConstantToImageKernel(S *output, const T *input, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input[idx] + C;
   }
}

template<class T, class S>
void AddConstantToImageKernelFunction(const T* input, S* output, unsigned int N, T C)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernel
   if (output == input)
     AddConstantToImageKernel <<< nBlocks, blockSize >>> (output, N, C);
   else
     AddConstantToImageKernel <<< nBlocks, blockSize >>> (output, input, N, C);

}

// template <typename S>
// struct addC
// {
//     const S a;

//     addC(S _a) : a(_a) {}

//     __host__ __device__
//         S operator()(const float& x) const { 
//             return a + x;
//         }
// };

// template<class T, class S>
// void AddConstantToImageKernelFunction(const T* input, S* output, unsigned int N, T C)
// {
//   thrust::device_ptr<const T> i1(input);
//   thrust::device_ptr<S> o1(output);
//   thrust::transform(i1, i1 + N, o1, addC<S>(C));

// }
// versions we wish to compile
#define THISTYPE float
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, unsigned int N, THISTYPE C);
#undef THISTYPE
#define THISTYPE int
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE short
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE unsigned char
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE


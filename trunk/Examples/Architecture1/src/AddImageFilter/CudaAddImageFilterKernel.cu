#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

// #include "thrust/functional.h"
// #include "thrust/transform.h"

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



// template <class T, class S>
// void AddImageKernelFunction(const T* input1, const T* input2, S* output, unsigned int N)
// {
//   if (input1 == output)
//     {
//     // not sure if this makes any difference
//     thrust::device_ptr<const T> i2(input2);
//     thrust::device_ptr<S> o1(output);
//     thrust::transform(o1, o1 + N, i2, o1, thrust::plus<S>());
//     }
//   else
//     {
//     thrust::device_ptr<const T> i1(input1);
//     thrust::device_ptr<const T> i2(input2);
//     thrust::device_ptr<S> o1(output);
//     thrust::transform(i1, i1 + N, i2, o1, thrust::plus<S>());
//     }

// }


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

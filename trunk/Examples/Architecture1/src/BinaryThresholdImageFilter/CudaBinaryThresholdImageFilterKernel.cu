#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T, class S>
__global__ void binaryThreshold(S *output, T lower, T upper, S inside, S outside, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      T A = output[idx];
      if ( lower <= A && A <= upper )
      {
      output[idx] = inside;
      }
      else
	{
	output[idx] = outside;
	}
   }
}

template <class T, class S>
__global__ void binaryThreshold(S *output, const T *input, T lower, T upper, S inside, S outside, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      T A = input[idx];
      if ( lower <= A && A <= upper )
      {
      output[idx] = inside;
      }
      else
	{
	output[idx] = outside;
	}
   }
}

template <class T, class S> 
void BinaryThresholdImageKernelFunction(const T* input, S* output, T m_LowerThreshold,
T m_UpperThreshold, S m_InsideValue, S m_OutsideValue, unsigned int N)
{
   // Compute execution configuration
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
   // Call  kernel
   if (output == input)
     binaryThreshold <<< nBlocks, blockSize >>> (output, m_LowerThreshold, m_UpperThreshold, m_InsideValue, m_OutsideValue, N);
   else
     binaryThreshold <<< nBlocks, blockSize >>> (output, input, m_LowerThreshold, m_UpperThreshold, m_InsideValue, m_OutsideValue, N);

}

// versions we wish to compile
#define THISTYPE float
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE m_LowerThreshold, THISTYPE m_UpperThreshold, THISTYPE m_InsideValue, THISTYPE m_OutsideValue, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE m_LowerThreshold, THISTYPE m_UpperThreshold, THISTYPE m_InsideValue, THISTYPE m_OutsideValue, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE m_LowerThreshold, THISTYPE m_UpperThreshold, THISTYPE m_InsideValue, THISTYPE m_OutsideValue, unsigned int N);

#undef THISTYPE

#define THISTYPE char
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE m_LowerThreshold, THISTYPE m_UpperThreshold, THISTYPE m_InsideValue, THISTYPE m_OutsideValue, unsigned int N);

#undef THISTYPE


#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T>
__global__ void MaxMinKernel(T *maxImage, T *minImage, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float offset = N/2;   

  for ( ; ((int)offset) > 0; )   
    {
    if (idx < offset)  {   		
    maxImage[idx] = (maxImage[idx] > maxImage[(int)(idx + offset)] ? maxImage[idx] : maxImage[(int)(idx + offset)]);
    minImage[idx] = (minImage[idx] < minImage[(int)(idx + offset)] ? minImage[idx] : minImage[(int)(idx + offset)]);
    } else {
    return;
    }
    offset /= 2;
    __syncthreads();
    }
}

template <class T>
__global__ void RescaleIntensityKernel(T *output, float offset, float factor, T max, T min, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
  if (idx < N)
    {
    T temp = output[idx] * factor + offset;
    output[idx] = (temp > max ? max : temp);
    output[idx] = (temp < min ? min : temp);
    }   
}

template <class T, class S>
__global__ void RescaleIntensityKernel(S *output, const T *input, float offset, float factor, T max, T min, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
  if (idx < N)
    {
    T temp = input[idx] * factor + offset;
    output[idx] = (temp > max ? max : temp);
    output[idx] = (temp < min ? min : temp);
    }   
}

template <class T, class S>
void CudaRescaleIntensityKernelFunction(const T* input, S* output, S outputMax, S outputMin, unsigned int N)
{
  T *maxImage, *minImage; 
   
  cudaMalloc(&maxImage, sizeof(T)*N);
  cudaMalloc(&minImage, sizeof(T)*N);
   
  cudaMemcpy(maxImage, input, sizeof(T)*N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(minImage, input, sizeof(T)*N, cudaMemcpyDeviceToDevice);

  // Compute execution configuration 
  int blockSize = 256;
  int nBlocks = N/(blockSize*2) + (N%(blockSize*2) == 0?0:1);

  // Call kernel
  MaxMinKernel <<< nBlocks, blockSize >>> (maxImage, minImage, N/2);
   
   
  T *inputMax, *inputMin;
  inputMax = (T*) malloc(sizeof(T));
  inputMin = (T*) malloc(sizeof(T));
   
  cudaMemcpy(inputMax, maxImage, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(inputMin, minImage, sizeof(T), cudaMemcpyDeviceToHost);
   
  float m_Factor = 0;
  float m_Offset = 0;
   
  if (inputMin[0] != inputMax[0])
    {
    m_Factor = (outputMax-outputMin) / (inputMax[0]-inputMin[0]);
    }
  else if (inputMax[0] != 0)
    {
    m_Factor = (outputMax-outputMin) / (inputMax[0]);
    }  		
  else
    {
    m_Factor = 0;
    }
   
  m_Offset = outputMin-inputMin[0] * m_Factor;
   
  nBlocks = N/(blockSize) + (N%blockSize == 0?0:1);

  if (input == output)
    RescaleIntensityKernel <<< nBlocks, blockSize >>> (output, m_Offset, m_Factor, outputMax, outputMin, N);
  else
    RescaleIntensityKernel <<< nBlocks, blockSize >>> (output, input, m_Offset, m_Factor, outputMax, outputMin, N);

  cudaFree(maxImage);
  cudaFree(minImage);
   
}

// versions we wish to compile
#define THISTYPE float
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE

#define THISTYPE char
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE


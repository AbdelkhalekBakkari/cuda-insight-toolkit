#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cutil.h>
#include "CudaStatisticsImageFilterKernel.h"

template <class T>
__global__ void StatisticsImageKernel(T *output, float* maxImage,
				      float* minImage, float* sumImage, 
				      float* sumOfSquaresImage) 
{

  int idx = threadIdx.x;

  for (float offset = blockDim.x / 2 ; offset >= 1; offset /= 2)
    {
    if (idx < offset)  {
    int next = (int)(idx + offset);
    maxImage[idx] = (maxImage[idx] > maxImage[next] ? maxImage[idx] : maxImage[next]);
    minImage[idx] = (minImage[idx] < minImage[next] ? minImage[idx] : minImage[next]);
    sumImage[idx] += sumImage[next];
    sumOfSquaresImage[idx] += sumOfSquaresImage[next];
    }
    __syncthreads();
    }

  if (blockIdx.x == 0)
    {
    for (float offset = gridDim.x / 2 ; offset >= 1; offset /= 2)
      {
      if (idx < offset)  {
      int next = (int)(idx + offset) * blockDim.x;
      maxImage[idx * blockDim.x] = (maxImage[idx * blockDim.x] > maxImage[next] ? maxImage[idx * blockDim.x] : maxImage[next]);
      minImage[idx * blockDim.x] = (minImage[idx * blockDim.x] < minImage[next] ? minImage[idx * blockDim.x] : minImage[next]);
      sumImage[idx * blockDim.x] += sumImage[next];
      sumOfSquaresImage[idx * blockDim.x] += sumOfSquaresImage[next];
      }
      offset /= 2;
      __syncthreads();
      }
    }
}
template <class T>
__global__ void StatisticsSquareImageKernel(T* sumOfSquaresImage, int N) 
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
    {
    sumOfSquaresImage[idx] *= sumOfSquaresImage[idx];
    }
}

template <class T> 
void StatisticsImageKernelFunction(const T* input, T* output,
				   StatisticsStruct * stats, unsigned int N) 
{

  float *maxImage, *minImage, *sumImage, *sumOfSquaresImage;

  cudaMalloc( &maxImage, sizeof(float) * N);
  cudaMalloc( &minImage, sizeof(float) * N);
  cudaMalloc( &sumImage, sizeof(float) * N);
  cudaMalloc( &sumOfSquaresImage, sizeof(float) * N);

  cudaMemcpy(maxImage, output, sizeof(float) * N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(minImage, output, sizeof(float) * N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(sumImage, output, sizeof(float) * N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(sumOfSquaresImage, output, sizeof(float) * N,
	     cudaMemcpyDeviceToDevice);

  // Compute execution configuration
  int blockSize = 128;
  int nBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);

  // Call kernel
  StatisticsSquareImageKernel <<< nBlocks, blockSize >>> (sumOfSquaresImage, N);
  cudaThreadSynchronize();
  StatisticsImageKernel <<< nBlocks, blockSize >>> (output, maxImage, minImage, sumImage, sumOfSquaresImage);

  float max, min, sum, sumOfSquares;

  cudaMemcpy(&sum, sumImage, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max, maxImage, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min, minImage, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sumOfSquares, sumOfSquaresImage, sizeof(float),
	     cudaMemcpyDeviceToHost);

  stats->Maximum = max;
  stats->Minimum = min;
  stats->Sum = sum;
  stats->Mean = sum / N;
  stats->Variance = (sumOfSquares - (sum * sum / N)) / (N - 1);
  stats->Sigma = sqrtf(stats->Variance);

  // Clean up
  cudaFree(maxImage);
  cudaFree(minImage);
  cudaFree(sumImage);
  cudaFree(sumOfSquaresImage);

}
// versions we wish to compile
#define THISTYPE float
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, StatisticsStruct * stats, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, StatisticsStruct * stats, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, StatisticsStruct * stats, unsigned int N);

#undef THISTYPE

#define THISTYPE char
template void BinaryThresholdImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, StatisticsStruct * stats, unsigned int N);

#undef THISTYPE

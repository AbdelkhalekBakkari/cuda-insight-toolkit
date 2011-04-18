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
__global__ void RescaleIntensityKernel(T *output, T offset, float factor, T max, T min, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (idx < N)
   {
	   T temp = output[idx] * factor + offset;
   		output[idx] = (temp > max ? max : temp);
   		output[idx] = (temp < min ? min : temp);
   }   
}

float * CudaRescaleIntensityKernelFunction(const float* input, unsigned int N, float outputMax, float outputMin)
{
   float *output = const_cast<float*>(input);
   float *maxImage, *minImage; 
   
   cudaMalloc((void **)&maxImage, sizeof(float)*N);
   cudaMalloc((void **)&minImage, sizeof(float)*N);
   
   cudaMemcpy(maxImage, output, sizeof(float)*N, cudaMemcpyDeviceToDevice);
   cudaMemcpy(minImage, output, sizeof(float)*N, cudaMemcpyDeviceToDevice);

   // Compute execution configuration 
   int blockSize = 256;
   int nBlocks = N/(blockSize*2) + (N%(blockSize*2) == 0?0:1);

   // Call kernel
   MaxMinKernel <<< nBlocks, blockSize >>> (maxImage, minImage, N/2);
   
   
   float *inputMax, *inputMin;
   inputMax = (float*) malloc(sizeof(float));
   inputMin = (float*) malloc(sizeof(float));
   
   cudaMemcpy(inputMax, maxImage, sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(inputMin, minImage, sizeof(float), cudaMemcpyDeviceToHost);
   
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
   RescaleIntensityKernel <<< nBlocks, blockSize >>> (output, m_Offset, m_Factor, outputMax, outputMin, N);
   
   cudaFree(maxImage);
   cudaFree(minImage);
   
   // Return pointer to the output
   return output;
}

/*
 * File Name:    cuda-kernel.h
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, January 18 2010, 10:00 
 * Last Modified: Thursday, January 14 2010, 15:58
 * 
 * File Description:
 *
 */
float * CudaNeighborhoodFilterKernelFunction(const float* input, unsigned int imageDimX, unsigned int imageDimY, unsigned int imageDimZ,
		unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ, unsigned int N);

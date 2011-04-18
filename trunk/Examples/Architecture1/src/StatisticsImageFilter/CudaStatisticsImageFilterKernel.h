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
struct StatisticsStruct
{
	float Count;
	float Minimum;
	float Maximum;
	float Mean;
	float Sigma;
	float Sum;
	float SumOfSquares;
	float Variance;
};

float * StatisticsImageKernelFunction(const float* input, StatisticsStruct* stats, unsigned int N);


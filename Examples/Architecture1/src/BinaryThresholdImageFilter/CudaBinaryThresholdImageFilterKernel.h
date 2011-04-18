/*
 * File Name:    cuda-kernel.h
 *
 * Author:        Phillip Ward
 * Creation Date: Wednesday, December 23 2009, 16:59 
 * Last Modified: Friday, January 15 2010, 15:18
 * 
 * File Description:
 *
 */
//__global__ void binaryThreshold(float *output,
//float lower, float upper, float inside, float outside, int N);

float* BinaryThreshold(const float* input, float m_LowerThreshold,
float m_UpperThreshold, float m_InsideValue, float m_OutsideValue,
unsigned int N);


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
template <class T, class S> extern
void SubtractConstantFromImageKernelFunction(const T* input1, S*output, unsigned int N, T C);


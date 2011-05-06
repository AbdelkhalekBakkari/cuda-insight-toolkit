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
void AddImageKernelFunction(const T* input1, const T* input2, S* output, unsigned int N);

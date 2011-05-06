/*
 * File Name:    CudaAddConstantToImageFilterKernel.h
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, January 18 2010, 10:00
 * Last Modified: Fri May  6 15:35:26 EST 2011
 *
 * File Description:
 *
 */
template <class T, class S> extern
void AddConstantToImageKernelFunction(const T* input1, S* output, unsigned int N, T C);


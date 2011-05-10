#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cutil.h>
#include "CudaStatisticsImageFilterKernel.h"
#include <limits>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>


 template <typename T>
 struct stats_square
 {
     __host__ __device__
	 float operator()(const T& x) const { 
       return (float)x * (float)x;
	 }
 };

 template <typename T>
 struct stats_cast
 {
     __host__ __device__
     float operator()(const T& x) const { 
       return (float)x;
     }
 };





 template <class T> 
 void StatisticsImageKernelFunction(const T* input, 
				    T &Minimum, T &Maximum, float &Sum, 
				    float &SumOfSquares, unsigned int N) 
 {
   thrust::device_ptr<const T> dptr(input);
   Maximum = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::min(), thrust::maximum<T>());

   Minimum = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::max(), thrust::minimum<T>());

   // using transform_reduce to include casting
   Sum = thrust::transform_reduce(dptr, dptr+N, stats_cast<T>(), 0, thrust::plus<float>());

   SumOfSquares = thrust::transform_reduce(dptr, dptr + N, stats_square<T>(), 0, thrust::plus<float>());
}
// versions we wish to compile
#define THISTYPE float
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE

#define THISTYPE unsigned char
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE

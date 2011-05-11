// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/generate.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <cstdlib>

int main(void)
{
  int *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);

  // generate random data on the host
  //thrust::host_vector<int> h_vec(100);
  //thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  //thrust::device_vector<int> d_vec(100);
  //int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}

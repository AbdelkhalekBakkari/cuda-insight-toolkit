
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
template <class T> extern
void StatisticsImageKernelFunction(const T* input, T *output, StatisticsStruct* stats, unsigned int N);


#ifndef __CudaMultiplyByConstantImageFilter_txx
#define __CudaMultiplyByConstantImageFilter_txx

#include "CudaMultiplyByConstantImageFilter.h"

#define CITK_OUT float
#define CITK_IN1 float
#include "CudaMultiplyByConstantImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaMultiplyByConstantImageFilter<TInputImage, TOutputImage>::CudaMultiplyByConstantImageFilter() {
	m_Constant = 1;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMultiplyByConstantImageFilter<TInputImage, TOutputImage>::PrintSelf(
		std::ostream& os, Indent indent) const {
	Superclass::PrintSelf(os, indent);

	os << indent << "Cuda MultiplyByConstant Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMultiplyByConstantImageFilter<TInputImage, TOutputImage>::GenerateData() 
{
  this->AllocateOutputs();
	// Set input and output type names.
	typename OutputImageType::Pointer output = this->GetOutput();
	typename InputImageType::ConstPointer input = this->GetInput();

	// Get Total Size
	const unsigned long N = input->GetPixelContainer()->Size();

	// Call Cu Function to execute kernel
	// Return pointer is to output array
	MultiplyByConstantImageKernelFunction<InputPixelType, OutputPixelType>
	  (input->GetDevicePointer(), output->GetDevicePointer(),
	   N, m_Constant);

}
}

#endif


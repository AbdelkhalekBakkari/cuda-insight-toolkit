#ifndef __CudaAbsImageFilter_txx
#define __CudaAbsImageFilter_txx

#include "CudaAbsImageFilter.h"
#include "CudaAbsImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaAbsImageFilter<TInputImage, TOutputImage>::CudaAbsImageFilter() {
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAbsImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream& os,
		Indent indent) const {
	Superclass::PrintSelf(os, indent);

	os << indent << "Cuda Abs Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAbsImageFilter<TInputImage, TOutputImage>::GenerateData() {
  this->AllocateOutputs();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();
  
  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();
  
  // Call Cuda Function
  AbsImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), 
							  output->GetDevicePointer(), N);
}
}

#endif


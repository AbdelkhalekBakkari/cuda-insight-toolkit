#ifndef __CudaAddConstantToImageFilter_txx
#define __CudaAddConstantToImageFilter_txx

#include "CudaAddConstantToImageFilter.h"

#include "CudaAddConstantToImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaAddConstantToImageFilter<TInputImage, TOutputImage>::CudaAddConstantToImageFilter() {
  m_Constant = 0;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAddConstantToImageFilter<TInputImage, TOutputImage>::PrintSelf(
  std::ostream& os, Indent indent) const {
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda AddConstantTo Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAddConstantToImageFilter<TInputImage, TOutputImage>::GenerateData() {
  this->AllocateOutputs();
  // Set input and output type names.
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();
  
  // Allocate Output Region
  // This code will set the output image to the same size as the input image.
  
  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();
  
  
  // Call Cu Function to execute kernel
  // Return pointer is to output array
  AddConstantToImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), 
								    output->GetDevicePointer(), N,
								    m_Constant);
  
}
}

#endif


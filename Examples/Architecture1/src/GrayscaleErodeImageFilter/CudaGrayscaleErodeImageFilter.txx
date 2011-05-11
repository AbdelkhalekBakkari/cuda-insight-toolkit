#ifndef __CudaGrayscaleErodeImageFilter_txx
#define __CudaGrayscaleErodeImageFilter_txx

#include "CudaGrayscaleErodeImageFilter.h"

#include "CudaGrayscaleErodeImageFilterKernel.h"

namespace itk
{

/*
    *
    */
template<class TInputImage, class TOutputImage, class TKernel>
CudaGrayscaleErodeImageFilter<TInputImage, TOutputImage, TKernel>::CudaGrayscaleErodeImageFilter()
{
}


/*
    *
    */
template <class TInputImage, class TOutputImage, class TKernel>
void CudaGrayscaleErodeImageFilter<TInputImage, TOutputImage, TKernel>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Filter" << std::endl;
}

/*
    *
    */
template <class TInputImage, class TOutputImage, class TKernel>
void CudaGrayscaleErodeImageFilter<TInputImage, TOutputImage, TKernel>
::GenerateData()
{
  this->AllocateOutputs();
  // Set input and output type names.
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();

  // Get Zero
  KernelPixelType zero = NumericTraits<KernelPixelType>::Zero;

  // Get Dimension
  const unsigned int D = input->GetLargestPossibleRegion().GetImageDimension();

  // Get Radius Dimensions
  const typename RadiusType::SizeValueType * radius = m_Kernel.GetRadius().GetSize();

  // Get Image Dimensions
  const typename SizeType::SizeValueType * imageDim = input->GetLargestPossibleRegion().GetSize().GetSize();

  // Get Kernel
  KernelPixelType* kernel = m_Kernel.GetBufferReference().begin();

  // Get Kernel Dimensions
  const typename KernelType::SizeType::SizeValueType * kernelDim = m_Kernel.GetSize().GetSize();

  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();
  
  CudaGrayscaleErodeImageFilterKernelFunction<InputPixelType, OutputPixelType, KernelPixelType>
    (input->GetDevicePointer(),
     output->GetDevicePointer(),
     imageDim, radius, kernel, kernelDim, zero, D, N);


}
}


#endif




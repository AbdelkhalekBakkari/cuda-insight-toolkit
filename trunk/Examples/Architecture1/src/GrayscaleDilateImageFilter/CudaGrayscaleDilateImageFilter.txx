#ifndef __CudaGrayscaleDilateImageFilter_txx
#define __CudaGrayscaleDilateImageFilter_txx

#include "CudaGrayscaleDilateImageFilter.h"

#include "CudaGrayscaleDilateImageFilterKernel.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage, class TKernel>
      CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>::CudaGrayscaleDilateImageFilter()
      {
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>
      ::GenerateData()
      {
         // Set input and output type names.
         typename OutputImageType::Pointer output = this->GetOutput();
         typename InputImageType::ConstPointer input = this->GetInput();

         // Allocate Output Region
         // This code will set the output image to the same size as the input image.
         typename OutputImageType::RegionType outputRegion;
         outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
         outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
         output->SetRegions(outputRegion);
         output->Allocate();

     	// Pointer for output array of output pixel type
     	typename TOutputImage::PixelType * ptr;

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

     	ptr = CudaGrayscaleDilateImageFilterKernelFunction(input->GetDevicePointer(),
     			imageDim, radius, kernel, kernelDim, zero, D, N);

     	// Set output array to output image

     	output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

     	//TInputImage * inputPtr = const_cast<TInputImage*> (this->GetInput());
     	//inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




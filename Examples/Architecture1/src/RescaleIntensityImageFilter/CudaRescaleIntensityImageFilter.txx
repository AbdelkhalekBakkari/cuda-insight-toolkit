#ifndef __CudaRescaleIntensityImageFilter_txx
#define __CudaRescaleIntensityImageFilter_txx

#include "CudaRescaleIntensityImageFilter.h"

#include "CudaRescaleIntensityImageFilterKernel.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage>
      CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>::CudaRescaleIntensityImageFilter()
      {
      	 m_OutputMaximum = NumericTraits<typename TOutputImage::PixelType>::max();
         m_OutputMinimum = NumericTraits<typename TOutputImage::PixelType>::NonpositiveMin();
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>
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

         // Get Total Size
         const unsigned long N = input->GetPixelContainer()->Size();

         // Pointer for output array of output pixel type
         typename TOutputImage::PixelType * ptr;

         // Call Cu Function to execute kernel
         // Return pointer is to output array
         ptr = CudaRescaleIntensityKernelFunction(input->GetDevicePointer(), N, m_OutputMaximum, m_OutputMinimum);

         // Set output array to output image
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

         TInputImage * inputPtr = const_cast<TInputImage*>(this->GetInput());
         inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




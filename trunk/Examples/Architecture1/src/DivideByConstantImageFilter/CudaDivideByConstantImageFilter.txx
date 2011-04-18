#ifndef __CudaDivideByConstantImageFilter_txx
#define __CudaDivideByConstantImageFilter_txx

#include "CudaDivideByConstantImageFilter.h"

#include "CudaDivideByConstantImageFilterKernel.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage>
      CudaDivideByConstantImageFilter<TInputImage, TOutputImage>::CudaDivideByConstantImageFilter()
      {
         m_Constant = 1;
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaDivideByConstantImageFilter<TInputImage, TOutputImage>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda DivideByConstant Image Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaDivideByConstantImageFilter<TInputImage, TOutputImage>
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

         const unsigned long N = input->GetPixelContainer()->Size();

         // Pointer for output array of output pixel type
         typename TOutputImage::PixelType * ptr;

         // Call Cu Function to execute kernel
         // Return pointer is to output array
         ptr = DivideByConstantImageKernelFunction(input->GetDevicePointer(), N, m_Constant);

         // Set output array to output image
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);


         // As CUDA output is stored in the same memory bank as the input
         // memory management must be turned off in the input.
         TInputImage * inputPtr = const_cast<TInputImage*>(this->GetInput());
         inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




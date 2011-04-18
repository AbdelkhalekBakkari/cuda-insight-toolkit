#ifndef __CudaBinaryThresholdImageFilter_txx
#define __CudaBinaryThresholdImageFilter_txx

#include "CudaBinaryThresholdImageFilter.h"

#include "CudaBinaryThresholdImageFilterKernel.h"

#include <vector>
#include <algorithm>

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage>
      CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>::CudaBinaryThresholdImageFilter()
      {
         m_InsideValue = NumericTraits<OutputPixelType>::max();
         m_OutsideValue = NumericTraits<OutputPixelType>::Zero;
         m_LowerThreshold = NumericTraits<InputPixelType>::NonpositiveMin();
         m_UpperThreshold = NumericTraits<InputPixelType>::max();
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Binary Threshold Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>
      ::GenerateData()
      {
         typename OutputImageType::Pointer output = this->GetOutput();
         typename InputImageType::ConstPointer input = this->GetInput();

         // Allocate Output Region
         typename OutputImageType::RegionType outputRegion;
         outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
         outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
         output->SetRegions(outputRegion);
         output->Allocate();

         // Get Total Size
         const unsigned long N = input->GetPixelContainer()->Size();

         // Call Cuda Function
         typename TOutputImage::PixelType * ptr;
         ptr = BinaryThreshold(input->GetDevicePointer(),
         m_LowerThreshold, m_UpperThreshold, m_InsideValue,
         m_OutsideValue, N);
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

         // Input must release control of cuda ptr since it is reused.
         TInputImage * inputPtr = const_cast<TInputImage*>(this->GetInput());
         inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




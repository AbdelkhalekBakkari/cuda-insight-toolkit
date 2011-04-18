#ifndef __CudaGrayscaleMorphologicalOpeningImageFilter_txx
#define __CudaGrayscaleMorphologicalOpeningImageFilter_txx

#include "CudaGrayscaleMorphologicalOpeningImageFilter.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage, class TKernel>
      CudaGrayscaleMorphologicalOpeningImageFilter<TInputImage, TOutputImage, TKernel>::CudaGrayscaleMorphologicalOpeningImageFilter()
      {
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleMorphologicalOpeningImageFilter<TInputImage, TOutputImage, TKernel>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleMorphologicalOpeningImageFilter<TInputImage, TOutputImage, TKernel>
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

         /** set up erosion and dilation methods */
         typename CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>::Pointer
           dilate = CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>::New();

         typename CudaGrayscaleErodeImageFilter<TOutputImage, TOutputImage, TKernel>::Pointer
           erode = CudaGrayscaleErodeImageFilter<TOutputImage, TOutputImage, TKernel>::New();

         dilate->SetKernel( this->GetKernel() );
         erode->SetKernel( this->GetKernel() );

         erode->SetInput( input );
         dilate->SetInput(  erode->GetOutput() );

         ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
         progress->SetMiniPipelineFilter(this);
         progress->RegisterInternalFilter(erode, .5f);
         progress->RegisterInternalFilter(dilate, .5f);

         /** execute the minipipeline */
         dilate->Update();

         // Pointer for output array of output pixel type
         const unsigned long N = dilate->GetOutput()->GetPixelContainer()->Size();
         typename TOutputImage::PixelType * ptr;
         ptr = dilate->GetOutput()->GetDevicePointer();
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);
         dilate->GetOutput()->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




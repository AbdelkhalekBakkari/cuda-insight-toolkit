#ifndef __CudaGrayscaleMorphologicalClosingImageFilter_txx
#define __CudaGrayscaleMorphologicalClosingImageFilter_txx

#include "CudaGrayscaleMorphologicalClosingImageFilter.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage, class TKernel>
      CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>::CudaGrayscaleMorphologicalClosingImageFilter()
      {
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage, class TKernel>
      void CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>
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

         dilate->SetInput( input );
         erode->SetInput(  dilate->GetOutput() );

         ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
         progress->SetMiniPipelineFilter(this);
         progress->RegisterInternalFilter(erode, .5f);
         progress->RegisterInternalFilter(dilate, .5f);

         dilate->SetInput( this->GetInput() );

         /** execute the minipipeline */
         erode->Update();

         // Pointer for output array of output pixel type
         const unsigned long N = erode->GetOutput()->GetPixelContainer()->Size();
         typename TOutputImage::PixelType * ptr;
         ptr = erode->GetOutput()->GetDevicePointer();
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);
         erode->GetOutput()->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif




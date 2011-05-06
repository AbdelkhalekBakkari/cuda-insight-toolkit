/*=========================================================================

  Program:   Cuda Insight Toolkit
  Module:    $RCSfile: itkCudaBinaryThresholdImageFilter.h,v $
  Language:  C++ & CUDA
  Date:      $Date: 2009-02-24 14:18:00 $
  Version:   $Revision: 1.0 $

  Copyright (c) 2010, Victorian Partnership for Advanced Computing
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

#ifndef __CudaBinaryThresholdImageFilter_h
#define __CudaBinaryThresholdImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaBinaryThresholdImageFilter
 *
 * \brief Binarize an input image by thresholding.
 *
 * This filter produces an output image whose pixels
 * are either one of two values ( OutsideValue or InsideValue ),
 * depending on whether the corresponding input image pixels
 * lie between the two thresholds ( LowerThreshold and UpperThreshold ).
 * Values equal to either threshold is considered to be between the thresholds.
 *
 * More precisely
 * \f[ Output(x_i) =
       \begin{cases}
         InsideValue  & \text{if $LowerThreshold \leq x_i \leq UpperThreshold$} \\
         OutsideValue & \text{otherwise}
        \end{cases}
   \f]
 *
 * This filter is templated over the input image type
 * and the output image type.
 *
 * The filter expect both images to have the same number of dimensions.
 *
 * The default values for LowerThreshold and UpperThreshold are:
 * LowerThreshold = NumericTraits<TInput>::NonpositiveMin();
 * UpperThreshold = NumericTraits<TInput>::max();
 * Therefore, generally only one of these needs to be set, depending
 * on whether the user wants to threshold above or below the desired threshold.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa CudaInPlaceImageFilter
 */


   template <class TInputImage, class TOutputImage>
      class ITK_EXPORT CudaBinaryThresholdImageFilter :
         public
         CudaInPlaceImageFilter<TInputImage, TOutputImage >
   {
      public:

         typedef TInputImage                 InputImageType;
         typedef TOutputImage                OutputImageType;


         typedef CudaBinaryThresholdImageFilter           Self;
         typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
                                             Superclass;
         typedef SmartPointer<Self>          Pointer;
         typedef SmartPointer<const Self>    ConstPointer;

         itkNewMacro(Self);

         itkTypeMacro(CudaBinaryThresholdImageFilter, CudaInPlaceImageFilter);

         typedef typename InputImageType::PixelType   InputPixelType;
         typedef typename OutputImageType::PixelType  OutputPixelType;

         typedef typename InputImageType::RegionType  InputImageRegionType;
         typedef typename OutputImageType::RegionType OutputImageRegionType;

         typedef typename InputImageType::SizeType    InputSizeType;

         itkSetMacro(LowerThreshold, InputPixelType);
         itkSetMacro(UpperThreshold, InputPixelType);
         itkSetMacro(InsideValue, OutputPixelType);
         itkSetMacro(OutsideValue, OutputPixelType);

         itkGetConstReferenceMacro(LowerThreshold, InputPixelType);
         itkGetConstReferenceMacro(UpperThreshold, InputPixelType);
         itkGetConstReferenceMacro(InsideValue, OutputPixelType);
         itkGetConstReferenceMacro(OutsideValue, OutputPixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck,
                  (Concept::EqualityComparable<OutputPixelType>));
  itkConceptMacro(InputPixelTypeComparable,
                  (Concept::Comparable<InputPixelType>));
  itkConceptMacro(InputOStreamWritableCheck,
                  (Concept::OStreamWritable<InputPixelType>));
  itkConceptMacro(OutputOStreamWritableCheck,
                  (Concept::OStreamWritable<OutputPixelType>));
  /** End concept checking */
#endif


      protected:
         CudaBinaryThresholdImageFilter();
         ~CudaBinaryThresholdImageFilter() {}
         void PrintSelf(std::ostream& os, Indent indent) const;
         void GenerateData();

      private:
         CudaBinaryThresholdImageFilter(const Self&);
         void operator=(const Self&);

         InputPixelType m_LowerThreshold;
         InputPixelType m_UpperThreshold;
         OutputPixelType m_InsideValue;
         OutputPixelType m_OutsideValue;
   };
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaBinaryThresholdImageFilter.txx"
#endif

#endif

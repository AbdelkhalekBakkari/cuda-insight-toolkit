/*=========================================================================

  Program:   Cuda Insight Toolkit
  Module:    $RCSfile: itkCudaRescaleIntensityImageFilter.h,v $
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


#ifndef __itkCudaRescaleIntensityImageFilter_h
#define __itkCudaRescaleIntensityImageFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{
  
/** \class CudaRescaleIntensityImageFilter
 * \brief Applies a linear transformation to the intensity levels of the
 * input Image.
 *
 * RescaleIntensityImageFilter applies pixel-wise a linear transformation
 * to the intensity values of input image pixels. The linear transformation
 * is defined by the user in terms of the minimum and maximum values that
 * the output image should have.
 *
 * All computations are performed in the precison of the input pixel's
 * RealType. Before assigning the computed value to the output pixel.
 *
 * NOTE: In this filter the minimum and maximum values of the input image are
 * computed internally using the MinimumMaximumImageCalculator. Users are not
 * supposed to set those values in this filter. If you need a filter where you
 * can set the minimum and maximum values of the input, please use the
 * IntensityWindowingImageFilter. If you want a filter that can use a
 * user-defined linear transformation for the intensity, then please use the
 * ShiftScaleImageFilter.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \sa ImageToImageFilter
 * 
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaRescaleIntensityImageFilter :
    public
ImageToImageFilter<TInputImage, TOutputImage >
{
public:

         typedef TInputImage                 InputImageType;
         typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaRescaleIntensityImageFilter  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage > 
                                             Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaRescaleIntensityImageFilter,
               ImageToImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;

  typedef typename NumericTraits<InputPixelType>::RealType RealType;

  itkSetMacro(OutputMaximum, OutputPixelType);
  itkSetMacro(OutputMinimum, OutputPixelType);

  itkGetConstReferenceMacro(OutputMaximum, OutputPixelType);
  itkGetConstReferenceMacro(OutputMinimum, OutputPixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputPixelType>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<OutputPixelType>));
  itkConceptMacro(RealTypeMultiplyOperatorCheck,
                  (Concept::MultiplyOperator<RealType>));
  itkConceptMacro(RealTypeAdditiveOperatorsCheck,
                  (Concept::AdditiveOperators<RealType>));
  /** End concept checking */
#endif

protected:
  CudaRescaleIntensityImageFilter();
  ~CudaRescaleIntensityImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaRescaleIntensityImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  OutputPixelType m_OutputMaximum;
  OutputPixelType m_OutputMinimum;

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaRescaleIntensityImageFilter.txx"
#endif

#endif

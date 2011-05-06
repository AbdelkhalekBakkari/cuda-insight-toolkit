/*=========================================================================

  Program:   Cuda Insight Toolkit
  Module:    $RCSfile: itkCudaMinimumImageFilter.h,v $
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


#ifndef __itkCudaMinimumImageFilter_h
#define __itkCudaMinimumImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{
  
/** \class CudaMinimumImageFilter
 * \brief Implements a pixel-wise operator Min(a,b) between two images.
 *
 * The pixel values of the output image are the minimum between the
 * corresponding pixels of the two input images.
 *
 * This class is parametrized over the types of the two
 * input images and the type of the output image.
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa CudaInPlaceImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMinimumImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

        typedef TInputImage                 InputImageType;
         typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaMinimumImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
                                             Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaMinimumImageFilter,
               CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;
  typedef typename OutputImageType::SizeType    OutputSizeType;

  void SetInput1( const TInputImage * image1 )
    {
    // Process object is not const-correct
    // so the const casting is required.
    SetNthInput(0, const_cast
        <TInputImage *>( image1 ));
    }

  void SetInput2( const TInputImage * image2 )
    {
    // Process object is not const-correct
    // so the const casting is required.
    SetNthInput(1, const_cast
        <TInputImage *>( image2 ));
    }

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(Input2ConvertibleToOutputCheck,
    (Concept::Convertible<typename TInputImage::PixelType,
                          typename TOutputImage::PixelType>));
  itkConceptMacro(Input1LessThanInput2Check,
    (Concept::LessThanComparable<typename TInputImage::PixelType,
                                 typename TInputImage::PixelType>));
  /** End concept checking */
#endif


protected:
  CudaMinimumImageFilter();
  ~CudaMinimumImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaMinimumImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMinimumImageFilter.txx"
#endif

#endif

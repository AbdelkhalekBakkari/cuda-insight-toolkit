/*=========================================================================

  Program:   Cuda Insight Toolkit
  Module:    $RCSfile: itkCudaDivideByConstantImageFilter.h,v $
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


#ifndef __itkCudaDivideByConstantImageFilter_h
#define __itkCudaDivideByConstantImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{
  
/** \class CudaDivideByConstantImageFilter
 *
 * \brief Divide input pixels by a constant.
 *
 * This filter is templated over the input image type
 * and the output image type.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 * \sa CudaInPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaDivideByConstantImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

        typedef TInputImage                 InputImageType;
         typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaDivideByConstantImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
                                             Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaDivideByConstantImageFilter,
               CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;
  typedef typename OutputImageType::SizeType    OutputSizeType;

  itkSetMacro(Constant, InputPixelType);
  itkGetConstReferenceMacro(Constant, InputPixelType);

  InputPixelType getConstant() const
  {
	  return m_Constant;
  }
#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputConvertibleToOutputCheck,
                   (Concept::Convertible<typename TInputImage::PixelType,
                    typename TOutputImage::PixelType>));
  itkConceptMacro(Input1Input2OutputDivideOperatorCheck,
                  (Concept::DivisionOperators<typename TInputImage::PixelType,
                   InputPixelType, 
                   typename TOutputImage::PixelType>));
  /** End concept checking */
#endif


protected:
  CudaDivideByConstantImageFilter();
  ~CudaDivideByConstantImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaDivideByConstantImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputPixelType m_Constant;

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaDivideByConstantImageFilter.txx"
#endif

#endif

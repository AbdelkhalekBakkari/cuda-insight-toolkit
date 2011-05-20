
#ifndef __itkCudaMultiplyByConstantImageFilter_h
#define __itkCudaMultiplyByConstantImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaMultiplyByConstantImageFilter
 *
 * \brief Multiply input pixels by a constant.
 *
 * This filter is templated over the input image type
 * and the output image type.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa CudaInPlaceImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMultiplyByConstantImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

        typedef TInputImage                 InputImageType;
         typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaMultiplyByConstantImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
                                             Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaMultiplyByConstantImageFilter,
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
  itkConceptMacro(Input1Input2OutputMultiplyOperatorCheck,
                  (Concept::MultiplyOperator<typename TInputImage::PixelType,
                   InputPixelType,
                   typename TOutputImage::PixelType>));
  /** End concept checking */
#endif


protected:
  CudaMultiplyByConstantImageFilter();
  ~CudaMultiplyByConstantImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaMultiplyByConstantImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputPixelType m_Constant;

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMultiplyByConstantImageFilter.txx"
#endif

#endif

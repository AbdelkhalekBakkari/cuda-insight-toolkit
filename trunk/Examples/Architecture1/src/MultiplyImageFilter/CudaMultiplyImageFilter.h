

#ifndef __itkCudaMultiplyImageFilter_h
#define __itkCudaMultiplyImageFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/** \class CudaMultiplyImageFilter
 * \brief Implements an operator for pixel-wise multiplication of two images.
 *
 * This class is parametrized over the types of the two
 * input images and the type of the output image.
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa ImageToImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMultiplyImageFilter :
    public
ImageToImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaMultiplyImageFilter  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaMultiplyImageFilter,
               ImageToImageFilter);

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
  itkConceptMacro(Input1Input2OutputMultiplyOperatorCheck,
		  (Concept::MultiplyOperator<typename TInputImage::PixelType,
		   typename TInputImage::PixelType,
		   typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  CudaMultiplyImageFilter();
  ~CudaMultiplyImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaMultiplyImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMultiplyImageFilter.txx"
#endif

#endif
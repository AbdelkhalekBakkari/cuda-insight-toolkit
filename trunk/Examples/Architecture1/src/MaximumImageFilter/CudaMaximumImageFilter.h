

#ifndef __itkCudaMaximumImageFilter_h
#define __itkCudaMaximumImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaMaximumImageFilter
 * \brief Implements a pixel-wise operator Max(a,b) between two images.
 *
 * The pixel values of the output image are the maximum between the
 * corresponding pixels of the two input images.
 *
 * This class is parametrized over the types of the two
 * input images and the type of the output image.
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing
 * (VPAC). Richard Beare
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa CudaInPlaceImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMaximumImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaMaximumImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaMaximumImageFilter,
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
  /** End concept checking */
#endif


protected:
  CudaMaximumImageFilter();
  ~CudaMaximumImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaMaximumImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMaximumImageFilter.txx"
#endif

#endif
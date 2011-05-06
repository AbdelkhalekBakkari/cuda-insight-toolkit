
#ifndef __itkCudaGrayscaleDilateImageFilter_h
#define __itkCudaGrayscaleDilateImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkNeighborhood.h"

namespace itk {

/**
 * \class CudaGrayscaleDilateImageFilter
 * \brief gray scale dilation of an image
 *
 * Dilate an image using grayscale morphology. Dilation takes the
 * maximum of all the pixels identified by the structuring element.
 *
 * The structuring element is assumed to be composed of binary
 * values (zero or one). Only elements of the structuring element
 * having values > 0 are candidates for affecting the center pixel.
 *
 * For the each input image pixel,
 *   - NeighborhoodIterator gives neighbors of the pixel.
 *   - Evaluate() member function returns the maximum value among
 *     the image neighbors where the kernel has elements > 0.
 *   - Replace the original value with the max value
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \sa ImageToImageFilter
 * \ingroup ImageEnhancement  MathematicalMorphologyImageFilters  CudaEnabled
 */


template<class TInputImage, class TOutputImage, class TKernel>
class ITK_EXPORT CudaGrayscaleDilateImageFilter: public ImageToImageFilter<TInputImage,
									   TOutputImage> {
public:

  /** Standard class typedefs. */
  typedef CudaGrayscaleDilateImageFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)	;

  /** Runtime information support. */
  itkTypeMacro(CudaGrayscaleDilateImageFilter,
	       ImageToImageFilter)	;

  typedef TInputImage                                   InputImageType;
  typedef TOutputImage                                  OutputImageType;
  typedef typename TInputImage::RegionType              RegionType;
  typedef typename TInputImage::SizeType                SizeType;
  typedef typename TInputImage::IndexType               IndexType;
  typedef typename TInputImage::PixelType               PixelType;
  typedef typename Superclass::OutputImageRegionType    OutputImageRegionType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Kernel typedef. */
  typedef TKernel KernelType;
  typedef typename TKernel::PixelType            KernelPixelType;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
		      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
		      TOutputImage::ImageDimension);
  itkStaticConstMacro(KernelDimension, unsigned int,
		      TKernel::NeighborhoodDimension);

  /** n-dimensional Kernel radius. */
  typedef typename KernelType::SizeType RadiusType;

  /** Set kernel (structuring element). */
  itkSetMacro(Kernel, KernelType);

  /** Get the kernel (structuring element). */
  itkGetConstReferenceMacro(Kernel, KernelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputConvertibleToOutputCheck,
		  (Concept::Convertible<PixelType, typename TOutputImage::PixelType>));
  itkConceptMacro(SameDimensionCheck1,
		  (Concept::SameDimension<InputImageDimension, OutputImageDimension>));
  itkConceptMacro(SameDimensionCheck2,
		  (Concept::SameDimension<InputImageDimension, KernelDimension>));
  itkConceptMacro(InputGreaterThanComparableCheck,
		  (Concept::GreaterThanComparable<PixelType>));
  itkConceptMacro(KernelGreaterThanComparableCheck,
		  (Concept::GreaterThanComparable<KernelPixelType>));
  /** End concept checking */
#endif

protected:
  CudaGrayscaleDilateImageFilter();
  ~CudaGrayscaleDilateImageFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaGrayscaleDilateImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** kernel or structuring element to use. */
  KernelType m_Kernel;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaGrayscaleDilateImageFilter.txx"
#endif

#endif

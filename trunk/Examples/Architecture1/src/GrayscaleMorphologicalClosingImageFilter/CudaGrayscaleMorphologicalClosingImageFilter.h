
#ifndef __itkCudaGrayscaleMorphologicalClosingImageFilter_h
#define __itkCudaGrayscaleMorphologicalClosingImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkProgressAccumulator.h"
#include "CudaGrayscaleDilateImageFilter.h"
#include "CudaGrayscaleErodeImageFilter.h"

namespace itk {

/**
 * \class CudaGrayscaleMorphologicalClosingImageFilter
 * \brief gray scale morphological closing of an image.
 *
 * This filter removes small (i.e., smaller than the structuring
 * element)holes and tube like structures in the interior or at the
 * boundaries of the image. The morphological closing of an image
 * "f" is defined as:
 * Closing(f) = Erosion(Dilation(f)).
 *
 * The structuring element is assumed to be composed of binary
 * values (zero or one). Only elements of the structuring element
 * having values > 0 are candidates for affecting the center pixel.
 *
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \sa ImageToImageFilter
 * \ingroup ImageEnhancement  MathematicalMorphologyImageFilters  CudaEnabled
 */


template<class TInputImage, class TOutputImage, class TKernel>
class ITK_EXPORT CudaGrayscaleMorphologicalClosingImageFilter: public ImageToImageFilter<TInputImage,
											 TOutputImage> {
public:

  /** Standard class typedefs. */
  typedef CudaGrayscaleMorphologicalClosingImageFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)	;

  /** Runtime information support. */
  itkTypeMacro(CudaGrayscaleMorphologicalClosingImageFilter,
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

  /** n-dimensional Kernel radius. */
  typedef typename KernelType::SizeType RadiusType;

  /** Set kernel (structuring element). */
  itkSetMacro(Kernel, KernelType);

  /** Get the kernel (structuring element). */
  itkGetConstReferenceMacro(Kernel, KernelType);

protected:
  CudaGrayscaleMorphologicalClosingImageFilter();
  ~CudaGrayscaleMorphologicalClosingImageFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaGrayscaleMorphologicalClosingImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** kernel or structuring element to use. */
  KernelType m_Kernel;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaGrayscaleMorphologicalClosingImageFilter.txx"
#endif

#endif
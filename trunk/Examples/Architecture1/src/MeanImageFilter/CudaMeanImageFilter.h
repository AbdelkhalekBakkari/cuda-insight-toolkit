/*=========================================================================

  Program:   Cuda Insight Toolkit
  Module:    $RCSfile: itkCudaMeanmImageFilter.h,v $
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


#ifndef __itkCudaMeanImageFilter_h
#define __itkCudaMeanImageFilter_h

#include "itkImageToImageFilter.h"

namespace itk {

/** \class CudaMeanImageFilter
 * \brief Applies an averaging filter to an image
 *
 * Computes an image where a given pixel is the mean value of the
 * the pixels in a neighborhood about the corresponding input pixel.
 *
 * A mean filter is one of the family of linear filters.
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \sa ImageToImageFilter
 * 
 * \ingroup IntensityImageFilters  CudaEnabled
 */


template<class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMeanImageFilter: public ImageToImageFilter<TInputImage,
		TOutputImage> {
public:

	typedef TInputImage InputImageType;
	typedef TOutputImage OutputImageType;

	/** Standard class typedefs. */
	typedef CudaMeanImageFilter Self;
	typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
	typedef SmartPointer<Self> Pointer;
	typedef SmartPointer<const Self> ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self)
	;

	/** Runtime information support. */
	itkTypeMacro(CudaMeanImageFilter,
			ImageToImageFilter)
	;

	typedef typename InputImageType::PixelType InputPixelType;
	typedef typename OutputImageType::PixelType OutputPixelType;

	typedef typename InputImageType::RegionType InputImageRegionType;
	typedef typename OutputImageType::RegionType OutputImageRegionType;

	typedef typename InputImageType::SizeType InputSizeType;

	/** Set the radius of the neighborhood used to compute the mean. */
	itkSetMacro(Radius, InputSizeType)
	;

	/** Get the radius of the neighborhood used to compute the mean */
	itkGetConstReferenceMacro(Radius, InputSizeType)
	;

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputPixelType>));
  /** End concept checking */
#endif

protected:
	CudaMeanImageFilter();
	~CudaMeanImageFilter() {
	}
	void PrintSelf(std::ostream& os, Indent indent) const;
	void GenerateData();

private:
	CudaMeanImageFilter(const Self&); //purposely not implemented
	void operator=(const Self&); //purposely not implemented

	InputSizeType m_Radius;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMeanImageFilter.txx"
#endif

#endif

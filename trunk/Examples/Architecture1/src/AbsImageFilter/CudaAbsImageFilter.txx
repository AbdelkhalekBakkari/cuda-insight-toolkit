#ifndef __CudaAbsImageFilter_txx
#define __CudaAbsImageFilter_txx

#include "CudaAbsImageFilter.h"
#include "CudaAbsImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaAbsImageFilter<TInputImage, TOutputImage>::CudaAbsImageFilter() {
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAbsImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream& os,
		Indent indent) const {
	Superclass::PrintSelf(os, indent);

	os << indent << "Cuda Abs Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaAbsImageFilter<TInputImage, TOutputImage>::GenerateData() {
	typename OutputImageType::Pointer output = this->GetOutput();
	typename InputImageType::ConstPointer input = this->GetInput();

	// Allocate Output Region
	typename OutputImageType::RegionType outputRegion;
	outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
	outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
	output->SetRegions(outputRegion);
	output->Allocate();

	// Get Total Size
	const unsigned long N = input->GetPixelContainer()->Size();

	// Call Cuda Function
	typename TOutputImage::PixelType * ptr;
	ptr = AbsImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), N);
	output->GetPixelContainer()->SetDevicePointer(ptr, N, true);
}
}

#endif


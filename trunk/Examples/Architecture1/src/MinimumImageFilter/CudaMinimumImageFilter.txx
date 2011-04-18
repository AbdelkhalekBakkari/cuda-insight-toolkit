#ifndef __CudaMinimumImageFilter_txx
#define __CudaMinimumImageFilter_txx

#include "CudaMinimumImageFilter.h"

#include "CudaMinimumImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaMinimumImageFilter<TInputImage, TOutputImage>::CudaMinimumImageFilter() {
	this->SetNumberOfRequiredInputs(2);
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMinimumImageFilter<TInputImage, TOutputImage>::PrintSelf(
		std::ostream& os, Indent indent) const {
	Superclass::PrintSelf(os, indent);

	os << indent << "Cuda Minimum Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMinimumImageFilter<TInputImage, TOutputImage>::GenerateData() {
	// Set input and output type names.
	typename OutputImageType::Pointer output = this->GetOutput();
	typename InputImageType::ConstPointer input1 = this->GetInput(0);
	typename InputImageType::ConstPointer input2 = this->GetInput(1);

	// Allocate Output Region
	// This code will set the output image to the same size as the input image.
	typename OutputImageType::RegionType outputRegion;
	outputRegion.SetSize(input1->GetLargestPossibleRegion().GetSize());
	outputRegion.SetIndex(input1->GetLargestPossibleRegion().GetIndex());
	output->SetRegions(outputRegion);
	output->Allocate();

	// Calculate number of Dimensions
	const unsigned long D =
			input1->GetLargestPossibleRegion().GetImageDimension();
	const unsigned long D2 =
			input2->GetLargestPossibleRegion().GetImageDimension();

	// Calculate size of array using number of dimensions.
	const unsigned long N = input1->GetPixelContainer()->Size();
	const unsigned long N2 = input2->GetPixelContainer()->Size();

	if (D != D2 || N != N2) {
		std::cerr << "Input Dimensions Dont Match" << std::endl;
		return;
	}

	// Pointer for output array of output pixel type
	typename TOutputImage::PixelType * ptr;

	// Call Cu Function to execute kernel
	// Return pointer is to output array
	ptr = MinimumImageKernelFunction(input1->GetDevicePointer(),
			input2->GetDevicePointer(), N);

	// Set output array to output image
	output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

	// As CUDA output is stored in the same memory bank as the input
	// memory management must be turned off in the input.
	TInputImage * inputPtr1 = const_cast<TInputImage*> (this->GetInput(1));
	inputPtr1->GetPixelContainer()->SetContainerManageDevice(false);
}
}

#endif


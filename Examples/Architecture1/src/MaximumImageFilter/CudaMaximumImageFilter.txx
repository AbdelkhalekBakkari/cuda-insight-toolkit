#ifndef __CudaMaximumImageFilter_txx
#define __CudaMaximumImageFilter_txx

#include "CudaMaximumImageFilter.h"

#include "CudaMaximumImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaMaximumImageFilter<TInputImage, TOutputImage>::CudaMaximumImageFilter() {
	this->SetNumberOfRequiredInputs(2);
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMaximumImageFilter<TInputImage, TOutputImage>::PrintSelf(
		std::ostream& os, Indent indent) const {
	Superclass::PrintSelf(os, indent);

	os << indent << "Cuda Maximum Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMaximumImageFilter<TInputImage, TOutputImage>::GenerateData() {
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
	const unsigned int D =
			input1->GetLargestPossibleRegion().GetImageDimension();
	const unsigned int D2 =
			input2->GetLargestPossibleRegion().GetImageDimension();

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
	ptr = MaximumImageKernelFunction(input1->GetDevicePointer(),
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


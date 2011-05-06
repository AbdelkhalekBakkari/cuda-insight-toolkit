#ifndef __CudaStatisticsImageFilter_txx
#define __CudaStatisticsImageFilter_txx

#include "CudaStatisticsImageFilter.h"

#include "CudaStatisticsImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage>
CudaStatisticsImageFilter<TInputImage>::CudaStatisticsImageFilter() 
{
  // first output is a copy of the image, DataObject created by
  // superclass
  //
  // allocate the data objects for the outputs which are
  // just decorators around pixel types
  for (int i = 1; i < 3; ++i) {
  typename PixelObjectType::Pointer
    output =
    static_cast<PixelObjectType*> (this->MakeOutput(i).GetPointer());
  this->ProcessObject::SetNthOutput(i, output.GetPointer());
  }

  // allocate the data objects for the outputs which are
  // just decorators around real types
  for (int i = 3; i < 7; ++i) {
  typename RealObjectType::Pointer output =
    static_cast<RealObjectType*> (this->MakeOutput(i).GetPointer());
  this->ProcessObject::SetNthOutput(i, output.GetPointer());
  }

  this->GetMinimumOutput()->Set(NumericTraits<PixelType>::max());
  this->GetMaximumOutput()->Set(NumericTraits<PixelType>::NonpositiveMin());
  this->GetMeanOutput()->Set(NumericTraits<RealType>::max());
  this->GetSigmaOutput()->Set(NumericTraits<RealType>::max());
  this->GetVarianceOutput()->Set(NumericTraits<RealType>::max());
  this->GetSumOutput()->Set(NumericTraits<RealType>::Zero);
}

template<class TInputImage>
DataObject::Pointer CudaStatisticsImageFilter<TInputImage>::MakeOutput(
  unsigned int output) 
{
  switch (output) {
  case 0:
    return static_cast<DataObject*> (TInputImage::New().GetPointer());
    break;
  case 1:
    return static_cast<DataObject*> (PixelObjectType::New().GetPointer());
    break;
  case 2:
    return static_cast<DataObject*> (PixelObjectType::New().GetPointer());
    break;
  case 3:
  case 4:
  case 5:
  case 6:
    return static_cast<DataObject*> (RealObjectType::New().GetPointer());
    break;
  default:
    // might as well make an image
    return static_cast<DataObject*> (TInputImage::New().GetPointer());
    break;
  }
}

/*
 *
 */
template<class TInputImage>
void CudaStatisticsImageFilter<TInputImage>::PrintSelf(std::ostream& os,
						       Indent indent) const {
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Statistics Filter" << std::endl;
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::PixelObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMinimumOutput() {
  return static_cast<PixelObjectType*> (this->ProcessObject::GetOutput(1));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::PixelObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMinimumOutput() const {
  return static_cast<const PixelObjectType*> (this->ProcessObject::GetOutput(
						1));
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::PixelObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMaximumOutput() {
  return static_cast<PixelObjectType*> (this->ProcessObject::GetOutput(2));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::PixelObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMaximumOutput() const {
  return static_cast<const PixelObjectType*> (this->ProcessObject::GetOutput(
						2));
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMeanOutput() {
  return static_cast<RealObjectType*> (this->ProcessObject::GetOutput(3));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetMeanOutput() const {
  return static_cast<const RealObjectType*> (this->ProcessObject::GetOutput(3));
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetSigmaOutput() {
  return static_cast<RealObjectType*> (this->ProcessObject::GetOutput(4));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetSigmaOutput() const {
  return static_cast<const RealObjectType*> (this->ProcessObject::GetOutput(4));
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetVarianceOutput() {
  return static_cast<RealObjectType*> (this->ProcessObject::GetOutput(5));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetVarianceOutput() const {
  return static_cast<const RealObjectType*> (this->ProcessObject::GetOutput(5));
}

template<class TInputImage>
typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetSumOutput() {
  return static_cast<RealObjectType*> (this->ProcessObject::GetOutput(6));
}

template<class TInputImage>
const typename CudaStatisticsImageFilter<TInputImage>::RealObjectType*
CudaStatisticsImageFilter<TInputImage>::GetSumOutput() const {
  return static_cast<const RealObjectType*> (this->ProcessObject::GetOutput(6));
}

template< class TInputImage >
void
CudaStatisticsImageFilter< TInputImage >
::AllocateOutputs()
{
  // Pass the input through as the output
  InputImagePointer image =
    const_cast< TInputImage * >( this->GetInput() );

  this->GraftOutput(image);

  // Nothing that needs to be allocated for the remaining outputs
}
/*
 *
 */
template<class TInputImage>
void CudaStatisticsImageFilter<TInputImage>::GenerateData() 
{
  this->AllocateOutputs();
  // Set input and output type names.
  typename InputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();

  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();

  StatisticsStruct * stats = new StatisticsStruct();

  stats->Count = NumericTraits<float>::Zero;
  stats->Sum = NumericTraits<float>::Zero;
  stats->SumOfSquares = NumericTraits<float>::Zero;
  stats->Minimum = NumericTraits<float>::max();
  stats->Maximum = NumericTraits<float>::NonpositiveMin();

  // Pointer for output array of output pixel type

  StatisticsImageKernelFunction<Input(output->GetDevicePtr(), stats, N);

  // Set output array to output image
  output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

  TInputImage * inputPtr = const_cast<TInputImage*> (this->GetInput());
  inputPtr->GetPixelContainer()->SetContainerManageDevice(false);

  this->GetMinimumOutput()->Set(stats->Minimum);
  this->GetMaximumOutput()->Set(stats->Maximum);
  this->GetMeanOutput()->Set(stats->Mean);
  this->GetSigmaOutput()->Set(stats->Sigma);
  this->GetVarianceOutput()->Set(stats->Variance);
  this->GetSumOutput()->Set(stats->Sum);
}

}
#endif


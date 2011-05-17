#include <stdio.h>
#include <stdlib.h>
#include "itkImage.h"
#include "CudaTest.h"

#include "CudaBinaryThresholdImageFilter.h"

using namespace std;

int main(int argc, char **argv)
{

  // Pixel Types
  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 2;
  bool InPlace = (bool)atoi(argv[3]);

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  typedef itk::CudaBinaryThresholdImageFilter<InputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetUpperThreshold(atof(argv[4]));
  filter->SetInsideValue(100);

  return(CudaTest1a<FilterType, InputImageType, OutputImageType>(InPlace, argv[1], argv[2], filter));
}




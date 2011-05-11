
#include <stdio.h>
#include <stdlib.h>
#include "itkImage.h"
#include "itkMeanImageFilter.h"
#include "CudaTest.h"

using namespace std;

int main(int argc, char **argv) {

  // Pixel Types
  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 2;
  int rad = atoi(argv[3]);

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  typedef itk::MeanImageFilter<InputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  InputImageType::SizeType radius;
  radius.Fill(rad);
  filter->SetRadius(radius);
  return(CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter));
}



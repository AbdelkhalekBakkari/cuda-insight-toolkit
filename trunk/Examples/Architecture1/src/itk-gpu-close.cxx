#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaTest.h"
#include "itkBinaryBallStructuringElement.h"
#include "CudaGrayscaleMorphologicalClosingImageFilter.h"

using namespace std;

#include <stdio.h>
#include <stdlib.h>

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
  // structuring element
  typedef itk::BinaryBallStructuringElement<unsigned char, Dimension> StructuringElementType;

  typedef itk::CudaGrayscaleMorphologicalClosingImageFilter<InputImageType, OutputImageType, StructuringElementType> FilterType;
  StructuringElementType  structuringElement;
  InputImageType::SizeType radius;
  radius.Fill(rad);
  
  structuringElement.SetRadius( rad );
  structuringElement.CreateStructuringElement();
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetKernel(structuringElement);
  return(CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter));
}


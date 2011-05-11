/*
 * File Name:    itk-gpu-close.cxx
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, December 21 2009, 14:15
 * Last Modified: Fri May  6 15:15:32 EST 2011
 *
 * File Description:
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaTest.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkGrayscaleMorphologicalOpeningImageFilter.h"

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

  typedef itk::GrayscaleMorphologicalOpeningImageFilter<InputImageType, OutputImageType, StructuringElementType> FilterType;
  StructuringElementType  structuringElement;
  InputImageType::SizeType radius;
  radius.Fill(rad);
  
  structuringElement.SetRadius( rad );
  structuringElement.CreateStructuringElement();
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetSafeBorder(false);
  filter->SetKernel(structuringElement);
  return(CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter));
}

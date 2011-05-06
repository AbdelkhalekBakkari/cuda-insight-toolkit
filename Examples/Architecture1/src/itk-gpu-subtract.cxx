/*
 * File Name:    itk-gpu-subtract.cxx
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, December 21 2009, 14:15 
 * Last Modified: Friday, January 15 2010, 16:35
 * 
 * File Description:
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaSubtractImageFilter.h"
#include "CudaTest.h"

using namespace std;

int main(int argc, char **argv) {
  int nFilters = atoi(argv[4]);
  bool InPlace = (bool)atoi(argv[5]);
  const unsigned Dimension = 2;
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::CudaSubtractImageFilter<InputImageType, OutputImageType> FilterType;

  return(CudaTest2<FilterType, InputImageType, OutputImageType>(nFilters, InPlace, argv[1], argv[2], argv[3]));
}


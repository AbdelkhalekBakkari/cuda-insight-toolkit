/*
 * File Name:    itk-gpu-multiply.cxx
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
#include "CudaMultiplyImageFilter.h"

using namespace std;

#include "CudaTest.h"



int main(int argc, char **argv) {
  int nFilters = atoi(argv[3]);
  bool InPlace = (bool)atoi(argv[4]);
  const unsigned Dimension = 2;
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::CudaMultiplyImageFilter<InputImageType, OutputImageType> FilterType;

  return(CudaTest2<FilterType, InputImageType, OutputImageType>(nFilters, InPlace, argv[1], argv[1], argv[2]));
}


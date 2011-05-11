/*
 * File Name:    itk-cpu-abs.cxx
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, December 21 2009, 14:15 
 * Last Modified: Fri May  6 15:07:46 EST 2011
 * 
 * File Description:
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "itkAbsImageFilter.h"
#include "CudaTest.h"


using namespace std;

int main(int argc, char **argv) {
  int nFilters = atoi(argv[3]);
  bool InPlace = (bool)atoi(argv[4]);
  const unsigned Dimension = 2;
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::AbsImageFilter<InputImageType, OutputImageType> FilterType;

  return(CudaTest1<FilterType, InputImageType, OutputImageType>(nFilters, InPlace, argv[1], argv[2]));
}


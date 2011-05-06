/*
 * File Name:    itk-gpu-statistics.cxx
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, December 21 2009, 14:15
 * Last Modified: Fri May  6 15:19:25 EST 2011
 *
 * File Description:
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaStatisticsImageFilter.h"
#include "CudaTests.h"

using namespace std;

int main(int argc, char **argv) 
{
  int nFilters = atoi(argv[3]);
  bool InPlace = (bool)atoi(argv[4]);
  const unsigned Dimension = 2;
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::CudaStatisticsImageFilter<InputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();

  int status = CudaTest1a<FilterType, InputImageType, OutputImageType>(nFilters, InPlace, argv[1], argv[2]);
  cout << "Statistic Output" << endl;
  cout << "Minimum: " << filter->GetMinimum() << endl;
  cout << "Maximum: " << filter->GetMaximum() << endl;
  cout << "Mean: " << filter->GetMean() << endl;
  cout << "Sigma: " << filter->GetSigma() << endl;
  cout << "Variance: " << filter->GetVariance() << endl;
  cout << "Sum: " << filter->GetSum() << endl;

  return(status);

}

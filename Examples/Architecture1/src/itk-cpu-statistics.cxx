/*
 * File Name:    itk-cpu-statistics.cxx
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
#include "itkStatisticsImageFilter.h"
#include "CudaTest.h"

using namespace std;

int main(int argc, char **argv) 
{
  const unsigned Dimension = 2;
  typedef unsigned char InputPixelType;
  typedef unsigned char OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::StatisticsImageFilter<InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();

  int status = CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter);
  cout << "Statistic Output" << endl;
  cout << "Minimum: " << (int)filter->GetMinimum() << endl;
  cout << "Maximum: " << (int)filter->GetMaximum() << endl;
  cout << "Mean: " << filter->GetMean() << endl;
  cout << "Sigma: " << filter->GetSigma() << endl;
  cout << "Variance: " << filter->GetVariance() << endl;
  cout << "Sum: " << filter->GetSum() << endl;

  return(status);

}

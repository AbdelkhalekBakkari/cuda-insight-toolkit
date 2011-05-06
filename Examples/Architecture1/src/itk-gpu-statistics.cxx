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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "CudaStatisticsImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "timer.h"

using namespace std;

int main(int argc, char **argv) {
  double start, end;

  // Pixel Types
  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 2;
  int nFilters = atoi(argv[3]);

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  typedef itk::CudaStatisticsImageFilter<InputImageType> FilterType;

  // Set Up Input File and Read Image
  ReaderType::Pointer reader1 = ReaderType::New();
  reader1->SetFileName(argv[1]);

  try {
  reader1->Update();
  } catch (itk::ExceptionObject exp) {
  cerr << "Reader caused problem." << endl;
  cerr << exp << endl;
  return 1;
  }

  for (unsigned int i = 0; i < 3; ++i) {
  if (i < Dimension) {
  cout
    << reader1->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
    << ", ";
  } else {
  cout << 1 << ", ";
  }
  }

  FilterType::Pointer filter[nFilters];
  filter[0] = FilterType::New();
  filter[0]->SetInput(reader1->GetOutput());

  for (int i = 1; i < nFilters; ++i) {
  filter[i] = FilterType::New();
  filter[i]->SetInput(filter[i - 1]->GetOutput());
  }

  try {
  start = getTime();
  filter[nFilters - 1]->Update();
  end = getTime();
  cout << end - start << endl;
  } catch (itk::ExceptionObject exp) {
  cerr << "Filter caused problem." << endl;
  cerr << exp << endl;
  return 1;
  }

  cout << argv[4] << endl;

  // Set Up Output File and Write Image
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[2]);
  writer->SetInput(filter[nFilters - 1]->GetOutput());

  try {
  writer->Update();
  } catch (itk::ExceptionObject exp) {
  cerr << "Filter caused problem." << endl;
  cerr << exp << endl;
  return 1;
  }

  cout << "Statistic Output" << endl;
  cout << "Minimum: " << filter[0]->GetMinimum() << endl;
  cout << "Maximum: " << filter[0]->GetMaximum() << endl;
  cout << "Mean: " << filter[0]->GetMean() << endl;
  cout << "Sigma: " << filter[0]->GetSigma() << endl;
  cout << "Variance: " << filter[0]->GetVariance() << endl;
  cout << "Sum: " << filter[0]->GetSum() << endl;

  return 0;
}


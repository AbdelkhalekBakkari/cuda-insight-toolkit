#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "itkImage.h"

#include "CudaSubtractConstantFromImageFilter.h"
#include "itkSubtractConstantFromImageFilter.h"
#include "CudaAddConstantToImageFilter.h"
#include "itkAddConstantToImageFilter.h"
#include "CudaStatisticsImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "timer.h"

#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

int main(int argc, char **argv) 
{

  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;


//  typedef itk::AddConstantToImageFilter<InputImageType, InputPixelType, InputImageType> AddType;
  typedef itk::CudaAddConstantToImageFilter<InputImageType, InputImageType> AddType;
//  typedef itk::CudaSubtractConstantFromImageFilter<InputImageType, InputImageType> SubType;
  typedef itk::SubtractConstantFromImageFilter<InputImageType, InputPixelType, InputImageType> SubType;
  {
  // dummy allocation to get the initialization out the way
  void *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);
  cudaFree(dInt);
  }

  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  // Set Up Input File and Read Image
//   ReaderType::Pointer reader1 = ReaderType::New();
//   reader1->SetFileName(argv[1]);

  InputImageType::Pointer input = InputImageType::New();
  InputImageType::SizeType size;
  InputImageType::IndexType start;

  size.Fill(500);
  start.Fill(0);

  InputImageType::RegionType reg;
  reg.SetSize(size);
  reg.SetIndex(start);

  input->SetRegions(reg);
  input->Allocate();
  input->FillBuffer(0);

//   input=reader1->GetOutput();
//   input->Update();
//   input->DisconnectPipeline();


  const unsigned iterations = 200;

  double startcpu=0, endcpu=0;
  double startgpu=0, endgpu=0;

  AddType::Pointer adder = AddType::New();
  adder->SetConstant(2);
  InputImageType::Pointer res;
  startgpu = getTime();


//  typedef itk::ImageFileWriter<OutputImageType> WriterType;
//  WriterType::Pointer writer = WriterType::New();
//   writer->SetFileName("res.nii.gz");
//   writer->SetInput(adder->GetOutput());
//   writer->Update();

#if 1

  for (unsigned i=0;i<iterations;i++)
    {
    adder->SetInput(input);
    adder->Modified();
    res = adder->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endgpu = getTime();
#endif
#if 1
  SubType::Pointer sub = SubType::New();
  sub->SetConstant(2);

  startcpu = getTime();
  for (unsigned i=0;i<iterations;i++)
    {
    sub->SetInput(input);
    sub->Modified();
    res = sub->GetOutput();
    res->Update();
    res->DisconnectPipeline();
//     std::cout << "A" << std::endl;
//     std::cout << input;
    input=res;
//     std::cout << "B" << std::endl;
//     std::cout << input;
    }
  endcpu = getTime();

  // check that everything has worked
#else
//   SubType::Pointer sub = SubType::New();
//   sub->SetConstant(2.1);
//   sub->SetInput(input);
//   sub->Modified();
//   sub->Update();
//   sub->SetInPlace(false);
//   res = sub->GetOutput();
//   std::cout << res;
//   res->Update();
//   res->DisconnectPipeline();
//   input=res;

#endif
  

#if 0
  typedef itk::StatisticsImageFilter<InputImageType> StatsType;
  StatsType::Pointer stats = StatsType::New();
//   stats->SetInput(sub->GetOutput());
  stats->SetInput(input);
  stats->Update();
  InputPixelType mx=stats->GetMaximum();

  std::cout << mx << " " << endgpu - startgpu << " " << endcpu - startcpu << std::endl;

  res=stats->GetOutput();
//   std::cout << input;
//   std::cout << res;
#endif
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("res.nii.gz");
  writer->SetInput(input);
 //  writer->SetInput(sub->GetOutput());
  writer->Update();
  std::cout << endgpu - startgpu << " " << endcpu - startcpu << std::endl;
  return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"

#include "CudaSubtractConstantFromImageFilter.h"
#include "itkAddConstantToImageFilter.h"
#include "CudaStatisticsImageFilter.h"
#include "timer.h"

int main(int argc, char **argv) 
{

  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;


  typedef itk::AddConstantToImageFilter<InputImageType, InputImageType> AddType;
  typedef itk::CudaSubtractConstantFromImageFilter<InputImageType, InputImageType> SubType;
  {
  // dummy allocation to get the initialization out the way
  int *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);
  cudaFree(dInt);
  }

  InputImageType::Pointer input = InputImageType::New();
  InputImageType::SizeType size;
  InputImageType::Index start;

  size.Fill(256);
  start.Fill(0);

  InputImageType::RegionType reg;
  reg.SetSize(size);
  reg.SetIndex(start);

  input->SetRegions(reg);
  input->Allocate();
  input->FillBuffer(0);

  const unsigned iterations = 100;

  double startcpu, endcpu;

  AddType::Pointer adder = AddType::New();
  adder->SetConstant(2.1);
  InputImageType::Pointer res;
  startcpu = getTime();
  for (unsigned i=0;i<iterations;i++)
    {
    adder->SetInput(input);
    adder->Modified();
    res = adder->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endcpu = getTime();

  SubType::Pointer sub = SubType::New();
  sub->SetConstant(2.1);

  double startgpu, endgpu;
  startgpu = getTime();
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
  // check that everything has worked


  typedef itk::CudaStatisticsImageFilter<InputImageType> StatsType;
  StatsType::Pointer stats = StatsType::New();
  stats->SetInput(input);
  stats->Update();
  InputPixelType mx=stats->GetMaximum();

  std::cout << mx << " " << endgpu - startgpu << " " << endcpu - startcpu << std::endl;

  return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "itkImage.h"

#include "CudaMeanImageFilter.h"
#include "itkMeanImageFilter.h"
#include "timer.h"

int main(int argc, char **argv) 
{

  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;



  typedef itk::CudaMeanImageFilter<InputImageType, InputImageType> CudaMeanType;
  typedef itk::MeanImageFilter<InputImageType, InputImageType> MeanType;
  {
  // dummy allocation to get the initialization out the way
  void *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);
  cudaFree(dInt);
  }


  InputImageType::Pointer input = InputImageType::New();
  InputImageType::SizeType size;
  InputImageType::IndexType start;

  size.Fill(250);
  start.Fill(0);

  InputImageType::RegionType reg;
  reg.SetSize(size);
  reg.SetIndex(start);

  input->SetRegions(reg);
  input->Allocate();
  input->FillBuffer(100);

  InputImageType::SizeType radius;
  radius.Fill(20);


  const unsigned iterations = 5;

  double startcpu=0, endcpu=0;
  double startgpu=0, endgpu=0;

  CudaMeanType::Pointer cmeanf = CudaMeanType::New();
  cmeanf->SetRadius(radius);

  MeanType::Pointer meanf = MeanType::New();
  meanf->SetRadius(radius);

  InputImageType::Pointer res;

  startgpu=getTime();
  for (unsigned i=0;i<iterations;i++)
    {
    cmeanf->SetInput(input);
    cmeanf->Modified();
    res = cmeanf->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endgpu = getTime();

  std::cout << "Finished GPU" << std::endl;

  startcpu = getTime();
  for (unsigned i=0;i<iterations;i++)
    {
    meanf->SetInput(input);
    meanf->Modified();
    res = meanf->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endcpu = getTime();

  std::cout << endgpu - startgpu << " " << endcpu - startcpu << std::endl;
  return EXIT_SUCCESS;
}

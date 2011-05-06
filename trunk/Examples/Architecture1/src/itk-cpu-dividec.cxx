/*
 * File Name:    myFirstITKFilter.cxx
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, December 21 2009, 14:15 
 * Last Modified: Friday, January 15 2010, 16:35
 * 
 * File Description:
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "itkDivideByConstantImageFilter.h"
#include "CudaTest.h"

using namespace std;

int main(int argc, char **argv) {

	// Pixel Types
	typedef float InputPixelType;
	typedef float OutputPixelType;
	const unsigned int Dimension = 2;
	bool InPlace = (bool)atoi(argv[4]);

	// IO Types
	// typedef itk::RGBPixel< InputPixelType >       PixelType;
	typedef itk::Image<InputPixelType, Dimension> InputImageType;
	typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
	typedef itk::ImageFileReader<InputImageType> ReaderType;
	typedef itk::ImageFileWriter<OutputImageType> WriterType;

	typedef itk::DivideByConstantImageFilter<InputImageType, InputPixelType, OutputImageType> FilterType;
	FilterType::Pointer filter = FilterType::New();
	filter->SetConstant(atof(argv[3]));
	return(CudaTest1a<FilterType, InputImageType, OutputImageType>(InPlace, argv[1], argv[2], filter));
}


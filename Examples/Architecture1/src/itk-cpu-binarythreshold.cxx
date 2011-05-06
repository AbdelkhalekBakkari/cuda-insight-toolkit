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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "timer.h"

using namespace std;

int main(int argc, char **argv) {
	double start, end;

	// Pixel Types
	typedef float InputPixelType;
	typedef float OutputPixelType;
	const unsigned int Dimension = 2;
	int nFilters = atoi(argv[3]);
	bool InPlace = (bool)atoi(argv[4]);


	// IO Types
	// typedef itk::RGBPixel< InputPixelType >       PixelType;
	typedef itk::Image<InputPixelType, Dimension> InputImageType;
	typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
	typedef itk::ImageFileReader<InputImageType> ReaderType;
	typedef itk::ImageFileWriter<OutputImageType> WriterType;

	typedef itk::BinaryThresholdImageFilter<InputImageType, OutputImageType> FilterType;

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

	filter[0]->SetUpperThreshold(atof(argv[5]));
	filter[0]->SetInsideValue(100);

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

	return 0;
}


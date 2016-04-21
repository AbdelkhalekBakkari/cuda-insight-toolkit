# Introduction #

This guide shows how to interface your [kernel](CUDA.md) with an [filter](ITK.md). This is often the most crucial step in developing a CUDA filter. Interfacing with ITK properly allows the filter to be used in the [pipeline](ITK.md). This is the standard use of a filter and the user expects this behavior. The most important feature that must be taken into account while interfacing is whether or not your filter is [In-Place] the step for handling this are below.

This guide assumes you have a basic knowledge of CUDA and will not go into further detail on it. If you are unfamiliar with CUDA a good place to start it is [Dr Dobb's](http://www.drdobbs.com/cpp/207200659). It also assumes you have a basic knowledge of ITK and the development of its filters. It is recommended that you read the [ITK Software Guide](http://www.itk.org/ItkSoftwareGuide.pdf).


# Guide #
## Step 1: Setting up the Filters ##

In order to write a CUDA filter in ITK you require 4 files:
  * A filter class header file (.h)
  * A filter class source file (.txx)
  * A kernel header file (.h)
  * A kernel source file (.cu)

The two filter class files should abide by [ITK's Style Guide](http://www.itk.org/Wiki/images/c/c6/ITKStyle.pdf) and be templated. The filter source file should include both the filter class header file and the kernel code header file.

The kernel source file should contain your CUDA kernel code. It should also have a c function to set up and launch the CUDA kernel code.  This function should take a global device pointer to the input image and return a global device pointer to the output image. This assumes your filter will produce an output image, in some cases it is better to return a structure containing the output values calculated such as in the StatisticsImageFilter.

## Step 2: Writing Kernel Function ##

An example of a c function to set up and launch a kernel where N is the number of pixels in the image:

```
         float* invertFunction(const float* input, unsigned int N)
         {
            // pointers
            float *output;
         
            // Allocate arrays on device
            cudaMalloc((void **) &output, sizeof(float)*N);
         
            // Compute execution configuration
            int blockSize = 128;
            int nBlocks = N/blockSize + (N%blockSize == 0 ? 0:1);
         
            // Execute kernel
            invertKernel <<< nBlocks, blockSize >>> (input, output, N);
         
            return output;
         }
```

The kernel header file should contain a pointer to the c function in your kernel source file. This is important for linking purposes.

An example of a kernel header file:

```
         float* invertFunction(const float* input, unsigned int N);
```

## Step 3: Writing the Kernel ##

Your CUDA kernel must be located in a .cu file.

An example of a kernel which assumes 255 is the highest value in the image:

```
         __global__ void invertKernel(const float *input, float *output, int N)
         {
             int idx = blockIdx.x * blockDim.x + threadIdx.x;
             if (idx<N)
             {
                 output[idx] = 255-input[idx];
             }
         }
```

## Step 4: Writing the ITK Filter ##

The filter header file should follow the standard ITK format including typedefs and macros. The Generate Data method in the filter source file is where your kernel will be referenced. Before calling the CUDA kernel a few things must be taken care of.

#### Set Up Output Image ####

The pipeline requires the output to be delivered in a different image object as the input image. For this reason we must tell the image what dimensions are image is in the Generate Data method.

```
         typename OutputImageType::RegionType outputRegion;
         outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
         outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
         output->SetRegions(outputRegion);
         output->Allocate();
```

#### Calculate Number of Pixels in Image ####

Most kernels will require the number of pixels in the image. This is to assist in the kernel execution configuration; particularly the number of threads and blocks required.

```
         const unsigned long N = input->GetPixelContainer()->Size();
```

#### Request Dimensions ####

Most CUDA filters cannot be templated to handle different dimensional images. The kernel required to compute a 2 dimensional image may be very different to a 3 dimensional image. You can retrieve the number of dimensions in an image like so:

```
     	const unsigned int D = input->GetLargestPossibleRegion().GetImageDimension();
```

#### Request Image Dimensions ####

Some more complicated filters, such as spatially aware filters, will require the dimensions of the image. This can be retrieved as a linear array like so:

```
     	const typename SizeType::SizeValueType * imageDim = input->GetLargestPossibleRegion().GetSize().GetSize();
```

#### Request Radius Dimensions ####

When doing a Neighborhood filter your CUDA kernel will require the dimensions of the user requested radius. This can be retrieved as a linear array like so:

```
         const typename RadiusType::SizeValueType * radius = m_Kernel.GetRadius().GetSize();
```

#### Request Structuring Element Kernel ####

More complicated filters, such as Morphological ones, will require a structuring element to run. A binary structuring element is an array equal to the radius dimensions, filled with 0s and 1s. The 1s correspond to values in a pixels neighborhood which should be included in the calculation. The following code retrieves the structuring element as a linear array.

```
         KernelPixelType* kernel = m_Kernel.GetBufferReference().begin();
```

_ITK refers to a Structuring Element as a Kernel._

#### Construct a Pointer for the Output Image ####

Your CUDA kernel function should return a pointer to the output image as a global device pointer. The only difficulty in holding this value is the template:

```
         typename TOutputImage::PixelType * ptr;
```

#### Retrieving the Input Image Pointer ####

The modified ITK architecture supplied by CUDA Insight Toolkit allows access to a pointer to the input image on the device. This pointer is to linear global memory. This pointer can be requested using the following code.

```
         input->GetDevicePointer();
```

#### Calling the Kernel Function ####

Once you have collected the values required to execute the filter they must be passed to the filter. They are done through the Kernel Function set up in Step 1. A simple function which takes an input image and the number of pixels in the image is shown below. This example stores the output image in a pointer similar to the one created earlier.

```
         ptr = invertFunction(input->GetDevicePointer(), N);
```

#### Handling the Output ####

After the filter has completed, the output pointer must be given to the output image to be passed down the pipeline. This can be done like so:

```
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);
```

In this example the 'ptr' is the pointer to output memory and the 'N' is the number of pixels in this image. The 'true' value is very important. This tells the output image to free the device memory once it is finished with it. **WARNING:** Setting this to false can lead to memory leaks on the device.

#### Handling the Input ####

The input is just as important. By default, this memory will be freed once your filter has finished executing. Sometimes this is not desirable. In the case of an [In-Place] filter, we want this memory to act as the output image. Freeing it would damage the output image. To stop the input image from freeing the device memory the following line of code must be executed.

```
         TInputImage * inputPtr = const_cast<TInputImage*>(this->GetInput());
         inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
```

**WARNING**: Do not use this code unless you know what you are doing. [In-Place] filters save a lot of time in memory allocation but are not always practical. When used incorrectly these filters can lead to memory leaks and interfere with device execution.
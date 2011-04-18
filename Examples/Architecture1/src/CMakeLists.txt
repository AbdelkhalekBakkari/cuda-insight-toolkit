
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

set(CUDA_PROPAGATE_HOST_FLAGS ON)

# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# The include_directories adds paths to both C/C++ compilation in the native
# compiler and cuda compilation in NVCC.  Note that CUDA_INCLUDE_DIRS is added
# automatically by CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.

INCLUDE_DIRECTORIES(
	${CUDA_INCLUDE_DIRS}
   AbsImageFilter
   AddImageFilter
   AddConstantToImageFilter
   BinaryThresholdImageFilter
   DivideImageFilter
   DivideByConstantImageFilter
   GrayscaleDilateImageFilter
   GrayscaleErodeImageFilter
   GrayscaleMorphologicalClosingImageFilter
   GrayscaleMorphologicalOpeningImageFilter
   MaximumImageFilter
   MeanImageFilter
   MinimumImageFilter
   MultiplyImageFilter
   MultiplyByConstantImageFilter
   NeighborhoodFilter
   RescaleIntensityImageFilter
   StatisticsImageFilter
   SubtractImageFilter
   SubtractConstantFromImageFilter
   CudaFunctions
)

##############################################################################
# There are four ways to compile source files with NVCC. 

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# Use one executable only.
CUDA_ADD_EXECUTABLE(ITK-GPU-OPEN 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-open.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-OPEN
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-CLOSE 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-close.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-CLOSE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-ERODE 
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-erode.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ERODE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-DILATE 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 itk-gpu-dilate.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-DILATE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-ABS 
 AbsImageFilter/CudaAbsImageFilterKernel.cu
 itk-gpu-abs.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ABS
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-ADD 
 AddImageFilter/CudaAddImageFilterKernel.cu
 itk-gpu-add.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ADD
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-BINARYT 
 BinaryThresholdImageFilter/CudaBinaryThresholdImageFilterKernel.cu
 itk-gpu-binarythreshold.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-BINARYT
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-DIVIDE 
 DivideImageFilter/CudaDivideImageFilterKernel.cu
 itk-gpu-divide.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-DIVIDE
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-MAXIMUM 
 MaximumImageFilter/CudaMaximumImageFilterKernel.cu
 itk-gpu-maximum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MAXIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-MINIMUM 
 MinimumImageFilter/CudaMinimumImageFilterKernel.cu
 itk-gpu-minimum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MINIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-MEAN 
 MeanImageFilter/CudaMeanImageFilterKernel.cu
 itk-gpu-mean.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MEAN
  ITKBasicFilters ITKCommon ITKIO
)
CUDA_ADD_EXECUTABLE(ITK-GPU-MULTIPLY 
 MultiplyImageFilter/CudaMultiplyImageFilterKernel.cu
 itk-gpu-multiply.cxx
 )
#CUDA_ADD_EXECUTABLE(ITK-GPU-MULTIPLY 
#itk-gpu-multiply.cu
#)
TARGET_LINK_LIBRARIES ( ITK-GPU-MULTIPLY
 ITKBasicFilters ITKCommon ITKIO
)

 CUDA_ADD_EXECUTABLE(ITK-GPU-RESCALE 
 RescaleIntensityImageFilter/CudaRescaleIntensityImageFilterKernel.cu
 itk-gpu-rescaleintensity.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-RESCALE
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-STATISTICS 
 StatisticsImageFilter/CudaStatisticsImageFilterKernel.cu
 itk-gpu-statistics.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-STATISTICS
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-SUBTRACT 
 SubtractImageFilter/CudaSubtractImageFilterKernel.cu
 itk-gpu-subtract.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-SUBTRACT
  ITKBasicFilters ITKCommon ITKIO
)
CUDA_ADD_EXECUTABLE(ITK-CPU-OPEN 
 itk-cpu-open.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-OPEN
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-CPU-CLOSE 
 itk-cpu-close.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-CLOSE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-ERODE 
 itk-cpu-erode.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ERODE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-DILATE 
 itk-cpu-dilate.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-DILATE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-ABS 
 itk-cpu-abs.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ABS
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-CPU-ADD 
 itk-cpu-add.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ADD
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-BINARYT 
 itk-cpu-binarythreshold.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-BINARYT
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-CPU-DIVIDE 
 itk-cpu-divide.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-DIVIDE
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-CPU-MAXIMUM 
 itk-cpu-maximum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MAXIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-MINIMUM 
 itk-cpu-minimum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MINIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-CPU-MEAN 
 itk-cpu-mean.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MEAN
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-CPU-MULTIPLY 
 itk-cpu-multiply.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MULTIPLY
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-CPU-RESCALE 
 itk-cpu-rescaleintensity.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-RESCALE
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-CPU-STATISTICS 
 itk-cpu-statistics.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-STATISTICS
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-CPU-SUBTRACT 
 itk-cpu-subtract.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-SUBTRACT
  ITKBasicFilters ITKCommon ITKIO
)
ADD_EXECUTABLE(ImageCompare ImageCompare.cxx)
TARGET_LINK_LIBRARIES(ImageCompare ITKCommon ITKIO)

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
#add_definitions(-DMULTIPLIER=2)
#
## You can set BUILD_SHARED_LIBS or you can pass STATIC, SHARED, or MODULE to
## CUDA_ADD_LIBRARY.  Remember that BUILD_SHARED_LIBS is only respected by
## CUDA_ADD_LIBRARY.  If you use CUDA_COMPILE or CUDA_WRAP_SRCS you must specify
## SHARED or STATIC as a parameter.
#set(BUILD_SHARED_LIBS ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#CUDA_ADD_LIBRARY(test_lib
#  test_lib.cu
#  external_dependency.h
#  # SHARED
#  # STATIC
#  OPTIONS "-DSTUFF=blah blah"
#  RELEASE --use_fast_math -DNDEBUG
#  DEBUG -g -DDEBUG
#  )
#
## Then link the shared library to another executable.
#ADD_EXECUTABLE(lib_example
#  main_for_lib.cc
#  )
#
## Specify the dependency.
#TARGET_LINK_LIBRARIES(lib_example
#  test_lib
#  )
#
## Using the CUDA_COMPILE macro
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
#set(source_files test_bin.cu)
#CUDA_COMPILE(CUDA_FILES test_bin.cu)
#
#ADD_EXECUTABLE(cuda_compile_example
#  ${CUDA_FILES}
#  ${source_files}
#  main.cc
#  external_dependency.h
#  )
#TARGET_LINK_LIBRARIES(cuda_compile_example
# ${CUDA_LIBRARIES}
# )
#
## Generating PTX files.
## CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)
#
## Add a special target to clean nvcc generated files.
#CUDA_BUILD_CLEAN_TARGET()
#

ADD_TEST(CPU-MULTIPLY ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MULTIPLY ${CMAKE_CURRENT_SOURCE_DIR}/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mult.mhd 1 
)

ADD_TEST(GPU-MULTIPLY ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MULTIPLY ${CMAKE_CURRENT_SOURCE_DIR}/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult.mhd 1 
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mult.mhd 
)


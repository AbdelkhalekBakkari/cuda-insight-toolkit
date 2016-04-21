# Requirements #

Before installing CITK, ensure you have the following things installed.
  * CUDA enabled graphics card
  * CUDA enabled drivers
  * CUDA toolkit

You can check to see whether you have a CUDA-capable GPU at the [NVIDIA CUDA website](http://www.nvidia.com/object/cuda_gpus.html).  The drivers and toolkit will also be available there, if needed.

# Step 1 - Extract ITK #

Download the latest release of ITK from [their website](http://www.itk.org).
Make a directory for CITK and extract the ITK files into it.

# Step 2 - Extract CITK #

Download the latest release of CITK from [here](http://code.google.com/p/cuda-insight-toolkit/downloads/list).
Extract CITK into the same directory as above. You will be prompted to overwrite several files, ensure you do so to avoid future errors.

# Step 3 - Build CITK #

Run CMake as you would with any regular build of ITK. Your CUDA directories should be detected automatically using the FindCUDA CMake Module, assuming you have met the requirements above. Ensure the USE\_CUDA\_SDK flag is off unless you have the SDK installed and plan to use it. Standard CITK does not require the SDK, and if you wish to use it for your own filters it is better to include it separately as needed.

# Step 4 - Set Up Project #

Step 4
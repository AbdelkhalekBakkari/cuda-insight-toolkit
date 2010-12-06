# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pward/Architecture1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pward/Architecture1

# Include any dependencies generated for this target.
include bin/CMakeFiles/ITK-GPU-ABS.dir/depend.make

# Include the progress variables for this target.
include bin/CMakeFiles/ITK-GPU-ABS.dir/progress.make

# Include the compile flags for this target's objects.
include bin/CMakeFiles/ITK-GPU-ABS.dir/flags.make

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o: bin/CMakeFiles/ITK-GPU-ABS.dir/flags.make
bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o: src/itk-gpu-abs.cxx
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pward/Architecture1/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o"
	cd /home/pward/Architecture1/bin && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o -c /home/pward/Architecture1/src/itk-gpu-abs.cxx

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.i"
	cd /home/pward/Architecture1/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/pward/Architecture1/src/itk-gpu-abs.cxx > CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.i

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.s"
	cd /home/pward/Architecture1/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/pward/Architecture1/src/itk-gpu-abs.cxx -o CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.s

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.requires:
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.requires

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.provides: bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.requires
	$(MAKE) -f bin/CMakeFiles/ITK-GPU-ABS.dir/build.make bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.provides.build
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.provides

bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.provides.build: bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.provides.build

bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: src/AbsImageFilter/CudaAbsImageFilterKernel.cu
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: src/AbsImageFilter/CudaAbsImageFilterKernel.cu
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /home/pward/NVIDIA_GPU_Computing_SDK/C/common/inc/cutil.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/_G_config.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/alloca.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/endian.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/pthreadtypes.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/select.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/sigset.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/stdio_lim.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/sys_errlist.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/time.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/types.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/typesizes.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/waitflags.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/waitstatus.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/wchar.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/bits/wordsize.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/endian.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/features.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/gconv.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/gnu/stubs-64.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/gnu/stubs.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/libio.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/stdio.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/stdlib.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/sys/cdefs.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/sys/select.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/sys/sysmacros.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/sys/types.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/time.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/wchar.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/include/xlocale.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/stdarg.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/stddef.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: /usr/local/cuda/include/cuda.h
bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o: bin/CMakeFiles/ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o.cmake
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pward/Architecture1/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o"
	cd /home/pward/Architecture1/bin && /usr/bin/cmake -E make_directory /home/pward/Architecture1/bin/.
	cd /home/pward/Architecture1/bin && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/pward/Architecture1/bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o -D generated_cubin_file:STRING=/home/pward/Architecture1/bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o.cubin.txt -P /home/pward/Architecture1/bin/CMakeFiles/ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o.cmake

# Object files for target ITK-GPU-ABS
ITK__GPU__ABS_OBJECTS = \
"CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o"

# External object files for target ITK-GPU-ABS
ITK__GPU__ABS_EXTERNAL_OBJECTS = \
"/home/pward/Architecture1/bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o"

bin/ITK-GPU-ABS: bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o
bin/ITK-GPU-ABS: /usr/local/cuda/lib64/libcudart.so
bin/ITK-GPU-ABS: /usr/lib64/libcuda.so
bin/ITK-GPU-ABS: /home/pward/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
bin/ITK-GPU-ABS: /usr/lib64/libuuid.so
bin/ITK-GPU-ABS: /usr/local/cuda/lib64/libcudart.so
bin/ITK-GPU-ABS: /usr/lib64/libcuda.so
bin/ITK-GPU-ABS: /home/pward/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
bin/ITK-GPU-ABS: bin/CMakeFiles/ITK-GPU-ABS.dir/build.make
bin/ITK-GPU-ABS: bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o
bin/ITK-GPU-ABS: bin/CMakeFiles/ITK-GPU-ABS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ITK-GPU-ABS"
	cd /home/pward/Architecture1/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ITK-GPU-ABS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/CMakeFiles/ITK-GPU-ABS.dir/build: bin/ITK-GPU-ABS
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/build

bin/CMakeFiles/ITK-GPU-ABS.dir/requires: bin/CMakeFiles/ITK-GPU-ABS.dir/itk-gpu-abs.cxx.o.requires
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/requires

bin/CMakeFiles/ITK-GPU-ABS.dir/clean:
	cd /home/pward/Architecture1/bin && $(CMAKE_COMMAND) -P CMakeFiles/ITK-GPU-ABS.dir/cmake_clean.cmake
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/clean

bin/CMakeFiles/ITK-GPU-ABS.dir/depend: bin/./ITK-GPU-ABS_generated_CudaAbsImageFilterKernel.cu.o
	cd /home/pward/Architecture1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pward/Architecture1 /home/pward/Architecture1/src /home/pward/Architecture1 /home/pward/Architecture1/bin /home/pward/Architecture1/bin/CMakeFiles/ITK-GPU-ABS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/CMakeFiles/ITK-GPU-ABS.dir/depend


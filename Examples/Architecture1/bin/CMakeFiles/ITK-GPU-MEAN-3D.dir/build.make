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
include bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/depend.make

# Include the progress variables for this target.
include bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/progress.make

# Include the compile flags for this target's objects.
include bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/flags.make

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/flags.make
bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o: src/itk-gpu-mean-3D.cxx
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pward/Architecture1/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o"
	cd /home/pward/Architecture1/bin && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o -c /home/pward/Architecture1/src/itk-gpu-mean-3D.cxx

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.i"
	cd /home/pward/Architecture1/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/pward/Architecture1/src/itk-gpu-mean-3D.cxx > CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.i

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.s"
	cd /home/pward/Architecture1/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/pward/Architecture1/src/itk-gpu-mean-3D.cxx -o CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.s

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.requires:
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.requires

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.provides: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.requires
	$(MAKE) -f bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/build.make bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.provides.build
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.provides

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.provides.build: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.provides.build

bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: src/MeanImageFilter/CudaMeanImageFilterKernel.cu
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: src/CudaFunctions/CudaNeighborhoodFunctions.cu
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: src/EclipseCompat.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: src/MeanImageFilter/CudaMeanImageFilterKernel.cu
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /home/pward/NVIDIA_GPU_Computing_SDK/C/common/inc/cutil.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/_G_config.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/alloca.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/confname.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/endian.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/environments.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/local_lim.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/locale.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/posix1_lim.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/posix2_lim.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/posix_opt.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/pthreadtypes.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/sched.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/select.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/setjmp.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/sigset.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/stdio_lim.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/sys_errlist.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/time.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/types.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/typesizes.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/waitflags.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/waitstatus.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/wchar.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/wordsize.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/bits/xopen_lim.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/algorithm
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/allocator.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/atomicity.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/basic_ios.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/basic_ios.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/basic_string.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/basic_string.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/char_traits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/codecvt.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/concept_check.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/cpp_type_traits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/functexcept.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/ios_base.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/istream.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/locale_classes.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/locale_facets.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/locale_facets.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/localefwd.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/ostream.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/postypes.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_algo.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_algobase.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_construct.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_function.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_heap.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_iterator.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_iterator_base_funcs.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_iterator_base_types.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_pair.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_raw_storage_iter.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_tempbuf.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stl_uninitialized.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/streambuf.tcc
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/streambuf_iterator.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/bits/stringfwd.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cctype
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/climits
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/clocale
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cstddef
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cstdio
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cstdlib
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cstring
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/ctime
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cwchar
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/cwctype
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/debug/debug.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/exception
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/exception_defines.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/ext/new_allocator.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/ios
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/iosfwd
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/iostream
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/istream
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/limits
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/locale
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/memory
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/new
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/ostream
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/streambuf
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/string
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/typeinfo
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/atomic_word.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/c++allocator.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/c++config.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/c++io.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/c++locale.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/cpu_defines.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/ctype_base.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/ctype_inline.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/gthr-default.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/gthr.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/messages_members.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/os_defines.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/c++/4.1.2/x86_64-redhat-linux/bits/time_members.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/ctype.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/endian.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/features.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/gconv.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/getopt.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/gnu/stubs-64.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/gnu/stubs.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/iconv.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/langinfo.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/libintl.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/libio.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/limits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/linux/limits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/locale.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/nl_types.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/pthread.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/sched.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/signal.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/stdint.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/stdio.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/stdlib.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/string.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/sys/cdefs.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/sys/select.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/sys/sysmacros.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/sys/types.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/time.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/unistd.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/wchar.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/wctype.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/include/xlocale.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/limits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/stdarg.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/stddef.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/syslimits.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: /usr/local/cuda/include/cuda.h
bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o: bin/CMakeFiles/ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o.cmake
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pward/Architecture1/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o"
	cd /home/pward/Architecture1/bin && /usr/bin/cmake -E make_directory /home/pward/Architecture1/bin/.
	cd /home/pward/Architecture1/bin && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/pward/Architecture1/bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o -D generated_cubin_file:STRING=/home/pward/Architecture1/bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o.cubin.txt -P /home/pward/Architecture1/bin/CMakeFiles/ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o.cmake

# Object files for target ITK-GPU-MEAN-3D
ITK__GPU__MEAN__3D_OBJECTS = \
"CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o"

# External object files for target ITK-GPU-MEAN-3D
ITK__GPU__MEAN__3D_EXTERNAL_OBJECTS = \
"/home/pward/Architecture1/bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o"

bin/ITK-GPU-MEAN-3D: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o
bin/ITK-GPU-MEAN-3D: /usr/local/cuda/lib64/libcudart.so
bin/ITK-GPU-MEAN-3D: /usr/lib64/libcuda.so
bin/ITK-GPU-MEAN-3D: /home/pward/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
bin/ITK-GPU-MEAN-3D: /usr/lib64/libuuid.so
bin/ITK-GPU-MEAN-3D: /usr/local/cuda/lib64/libcudart.so
bin/ITK-GPU-MEAN-3D: /usr/lib64/libcuda.so
bin/ITK-GPU-MEAN-3D: /home/pward/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
bin/ITK-GPU-MEAN-3D: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/build.make
bin/ITK-GPU-MEAN-3D: bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o
bin/ITK-GPU-MEAN-3D: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ITK-GPU-MEAN-3D"
	cd /home/pward/Architecture1/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ITK-GPU-MEAN-3D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/build: bin/ITK-GPU-MEAN-3D
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/build

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/requires: bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/itk-gpu-mean-3D.cxx.o.requires
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/requires

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/clean:
	cd /home/pward/Architecture1/bin && $(CMAKE_COMMAND) -P CMakeFiles/ITK-GPU-MEAN-3D.dir/cmake_clean.cmake
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/clean

bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/depend: bin/./ITK-GPU-MEAN-3D_generated_CudaMeanImageFilterKernel.cu.o
	cd /home/pward/Architecture1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pward/Architecture1 /home/pward/Architecture1/src /home/pward/Architecture1 /home/pward/Architecture1/bin /home/pward/Architecture1/bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/CMakeFiles/ITK-GPU-MEAN-3D.dir/depend


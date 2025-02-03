# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr

# Include any dependencies generated for this target.
include CMakeFiles/lab1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lab1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lab1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab1.dir/flags.make

CMakeFiles/lab1.dir/lab1.cu.o: CMakeFiles/lab1.dir/flags.make
CMakeFiles/lab1.dir/lab1.cu.o: CMakeFiles/lab1.dir/includes_CUDA.rsp
CMakeFiles/lab1.dir/lab1.cu.o: lab1.cu
CMakeFiles/lab1.dir/lab1.cu.o: CMakeFiles/lab1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lab1.dir/lab1.cu.o"
	/apps/rocky9/cuda/12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/lab1.dir/lab1.cu.o -MF CMakeFiles/lab1.dir/lab1.cu.o.d -x cu -c /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/lab1.cu -o CMakeFiles/lab1.dir/lab1.cu.o

CMakeFiles/lab1.dir/lab1.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lab1.dir/lab1.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/lab1.dir/lab1.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lab1.dir/lab1.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lab1
lab1_OBJECTS = \
"CMakeFiles/lab1.dir/lab1.cu.o"

# External object files for target lab1
lab1_EXTERNAL_OBJECTS =

lab1: CMakeFiles/lab1.dir/lab1.cu.o
lab1: CMakeFiles/lab1.dir/build.make
lab1: src/libcudaLib.a
lab1: src/libcpuLib.a
lab1: /apps/rocky9/cuda/12.6/lib64/libcudart_static.a
lab1: /usr/lib64/librt.a
lab1: /apps/rocky9/cuda/12.6/lib64/libcudart_static.a
lab1: /usr/lib64/librt.a
lab1: CMakeFiles/lab1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lab1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab1.dir/build: lab1
.PHONY : CMakeFiles/lab1.dir/build

CMakeFiles/lab1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab1.dir/clean

CMakeFiles/lab1.dir/depend:
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/CMakeFiles/lab1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab1.dir/depend


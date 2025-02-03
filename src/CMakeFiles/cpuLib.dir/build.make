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
include src/CMakeFiles/cpuLib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/cpuLib.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cpuLib.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cpuLib.dir/flags.make

src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o: src/CMakeFiles/cpuLib.dir/flags.make
src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o: src/cpuLib.cpp
src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o: src/CMakeFiles/cpuLib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o"
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o -MF CMakeFiles/cpuLib.dir/cpuLib.cpp.o.d -o CMakeFiles/cpuLib.dir/cpuLib.cpp.o -c /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src/cpuLib.cpp

src/CMakeFiles/cpuLib.dir/cpuLib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpuLib.dir/cpuLib.cpp.i"
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src/cpuLib.cpp > CMakeFiles/cpuLib.dir/cpuLib.cpp.i

src/CMakeFiles/cpuLib.dir/cpuLib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpuLib.dir/cpuLib.cpp.s"
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src/cpuLib.cpp -o CMakeFiles/cpuLib.dir/cpuLib.cpp.s

# Object files for target cpuLib
cpuLib_OBJECTS = \
"CMakeFiles/cpuLib.dir/cpuLib.cpp.o"

# External object files for target cpuLib
cpuLib_EXTERNAL_OBJECTS =

src/libcpuLib.a: src/CMakeFiles/cpuLib.dir/cpuLib.cpp.o
src/libcpuLib.a: src/CMakeFiles/cpuLib.dir/build.make
src/libcpuLib.a: src/CMakeFiles/cpuLib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcpuLib.a"
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && $(CMAKE_COMMAND) -P CMakeFiles/cpuLib.dir/cmake_clean_target.cmake
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpuLib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cpuLib.dir/build: src/libcpuLib.a
.PHONY : src/CMakeFiles/cpuLib.dir/build

src/CMakeFiles/cpuLib.dir/clean:
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src && $(CMAKE_COMMAND) -P CMakeFiles/cpuLib.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cpuLib.dir/clean

src/CMakeFiles/cpuLib.dir/depend:
	cd /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src /home/gruber13/build/lab1/cuda-assignment-1-SilviuGr/src/CMakeFiles/cpuLib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cpuLib.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
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

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/a100/workspace/Mithril_MultiGPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/a100/workspace/Mithril_MultiGPU/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_full_structual_graph.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_full_structual_graph.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_full_structual_graph.dir/flags.make

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o: tests/CMakeFiles/test_full_structual_graph.dir/flags.make
tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o: ../tests/test_full_structual_graph.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/a100/workspace/Mithril_MultiGPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o"
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o -c /home/a100/workspace/Mithril_MultiGPU/tests/test_full_structual_graph.cc

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.i"
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/a100/workspace/Mithril_MultiGPU/tests/test_full_structual_graph.cc > CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.i

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.s"
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/a100/workspace/Mithril_MultiGPU/tests/test_full_structual_graph.cc -o CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.s

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.requires:

.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.requires

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.provides: tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.requires
	$(MAKE) -f tests/CMakeFiles/test_full_structual_graph.dir/build.make tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.provides.build
.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.provides

tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.provides.build: tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o


# Object files for target test_full_structual_graph
test_full_structual_graph_OBJECTS = \
"CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o"

# External object files for target test_full_structual_graph
test_full_structual_graph_EXTERNAL_OBJECTS =

tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: tests/CMakeFiles/test_full_structual_graph.dir/build.make
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: libcore.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: libcontext.a
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: libparallel.a
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: libcudahelp.a
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/mpi/lib/libmpi.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/boost-install/lib/libboost_program_options.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/cu111/lib64/libcudart.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/cu111/lib64/libcublas.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/cu111/lib64/libcudnn.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/cu111/lib64/libcusparse.so
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: /home/a100/nccl-install/lib/libnccl_static.a
tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o: tests/CMakeFiles/test_full_structual_graph.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/a100/workspace/Mithril_MultiGPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o"
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_full_structual_graph.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_full_structual_graph.dir/build: tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o

.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/build

# Object files for target test_full_structual_graph
test_full_structual_graph_OBJECTS = \
"CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o"

# External object files for target test_full_structual_graph
test_full_structual_graph_EXTERNAL_OBJECTS =

tests/test_full_structual_graph: tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o
tests/test_full_structual_graph: tests/CMakeFiles/test_full_structual_graph.dir/build.make
tests/test_full_structual_graph: libcore.so
tests/test_full_structual_graph: libcontext.a
tests/test_full_structual_graph: libparallel.a
tests/test_full_structual_graph: libcudahelp.a
tests/test_full_structual_graph: /home/a100/mpi/lib/libmpi.so
tests/test_full_structual_graph: /home/a100/boost-install/lib/libboost_program_options.so
tests/test_full_structual_graph: /home/a100/cu111/lib64/libcudart.so
tests/test_full_structual_graph: /home/a100/cu111/lib64/libcublas.so
tests/test_full_structual_graph: /home/a100/cu111/lib64/libcudnn.so
tests/test_full_structual_graph: /home/a100/cu111/lib64/libcusparse.so
tests/test_full_structual_graph: /home/a100/nccl-install/lib/libnccl_static.a
tests/test_full_structual_graph: tests/CMakeFiles/test_full_structual_graph.dir/cmake_device_link.o
tests/test_full_structual_graph: tests/CMakeFiles/test_full_structual_graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/a100/workspace/Mithril_MultiGPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_full_structual_graph"
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_full_structual_graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_full_structual_graph.dir/build: tests/test_full_structual_graph

.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/build

tests/CMakeFiles/test_full_structual_graph.dir/requires: tests/CMakeFiles/test_full_structual_graph.dir/test_full_structual_graph.cc.o.requires

.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/requires

tests/CMakeFiles/test_full_structual_graph.dir/clean:
	cd /home/a100/workspace/Mithril_MultiGPU/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_full_structual_graph.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/clean

tests/CMakeFiles/test_full_structual_graph.dir/depend:
	cd /home/a100/workspace/Mithril_MultiGPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/a100/workspace/Mithril_MultiGPU /home/a100/workspace/Mithril_MultiGPU/tests /home/a100/workspace/Mithril_MultiGPU/build /home/a100/workspace/Mithril_MultiGPU/build/tests /home/a100/workspace/Mithril_MultiGPU/build/tests/CMakeFiles/test_full_structual_graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_full_structual_graph.dir/depend


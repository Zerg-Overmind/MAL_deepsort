# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/extras/cppapi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/extras/cppapi/build

# Include any dependencies generated for this target.
include CMakeFiles/infer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/infer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/infer.dir/flags.make

CMakeFiles/infer.dir/infer.cpp.o: CMakeFiles/infer.dir/flags.make
CMakeFiles/infer.dir/infer.cpp.o: ../infer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/extras/cppapi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/infer.dir/infer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/infer.dir/infer.cpp.o -c /workspace/extras/cppapi/infer.cpp

CMakeFiles/infer.dir/infer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/infer.dir/infer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/extras/cppapi/infer.cpp > CMakeFiles/infer.dir/infer.cpp.i

CMakeFiles/infer.dir/infer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/infer.dir/infer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/extras/cppapi/infer.cpp -o CMakeFiles/infer.dir/infer.cpp.s

# Object files for target infer
infer_OBJECTS = \
"CMakeFiles/infer.dir/infer.cpp.o"

# External object files for target infer
infer_EXTERNAL_OBJECTS =

infer: CMakeFiles/infer.dir/infer.cpp.o
infer: CMakeFiles/infer.dir/build.make
infer: libretinanet.so
infer: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.4.0
infer: /usr/local/cuda/lib64/libcudart_static.a
infer: /usr/lib/x86_64-linux-gnu/librt.so
infer: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.4.0
infer: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.4.0
infer: CMakeFiles/infer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/extras/cppapi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable infer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/infer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/infer.dir/build: infer

.PHONY : CMakeFiles/infer.dir/build

CMakeFiles/infer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/infer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/infer.dir/clean

CMakeFiles/infer.dir/depend:
	cd /workspace/extras/cppapi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/extras/cppapi /workspace/extras/cppapi /workspace/extras/cppapi/build /workspace/extras/cppapi/build /workspace/extras/cppapi/build/CMakeFiles/infer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/infer.dir/depend


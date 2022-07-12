# ==================================================================================================
# This file is part of the cuBLASt project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================
#
# Defines the following variables:
#   CUBLAS_FOUND          Boolean holding whether or not the cuBLAS library was found
#   CUBLAS_INCLUDE_DIRS   The CUDA and cuBLAS include directory
#   CUDA_LIBRARIES        The CUDA library
#   CUBLAS_LIBRARIES      The cuBLAS library
#
# In case CUDA is not installed in the default directory, set the CUDA_ROOT variable to point to
# the root of cuBLAS, such that 'cublas_v2.h' can be found in $CUDA_ROOT/include. This can either be
# done using an environmental variable (e.g. export CUDA_ROOT=/path/to/cuBLAS) or using a CMake
# variable (e.g. cmake -DCUDA_ROOT=/path/to/cuBLAS ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CUDNN_HINTS
  /home/p100/cuda112
)
set(CUDNN_PATHS
  /home/p100/cuda112
)

# Finds the include directories
find_path(CUDNN_INCLUDE_DIRS
  NAMES cudnn.h cuda.h
  HINTS ${CUDNN_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CUDNN_PATHS}
  DOC "cuDNN include header cudnn.h"
)
mark_as_advanced(CUDNN_INCLUDE_DIRS)

# Finds the libraries
#find_library(CUDA_LIBRARIES
 # NAMES cudart
 # HINTS ${CUDNN_HINTS}
  #PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  #PATHS ${CUDNN_PATHS}
  #DOC "CUDA library"
#)
#mark_as_advanced(CUDA_LIBRARIES)
find_library(CUDNN_LIBRARIES
  NAMES cudnn
  HINTS ${CUDNN_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUDNN_PATHS}
  DOC "cuDNN library"
)
mark_as_advanced(CUDNN_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CUDNN_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'cudnn.h', install CUDA/cuDNN or set CUDA_ROOT")
endif()
#if(NOT CUDA_LIBRARIES)
 #   message(STATUS "Could NOT find CUDA library, install it or set CUDA_ROOT")
#endif()
if(NOT CUDNN_LIBRARIES)
    message(STATUS "Could NOT find cuDNN library, install it or set CUDA_ROOT")
endif()

# Determines whether or not cuDNN was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN DEFAULT_MSG CUDNN_INCLUDE_DIRS CUDA_LIBRARIES CUDNN_LIBRARIES)

# ===============================================================

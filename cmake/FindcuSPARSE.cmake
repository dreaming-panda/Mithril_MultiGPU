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
set(CUSPARSE_HINTS
  /opt/apps/cuda/11.3/
  /usr/local/cuda-11.6/
  /usr/local/cuda
)
set(CUSPARSE_PATHS
  /opt/apps/cuda/11.3/
)

# Finds the include directories
find_path(CUSPARSE_INCLUDE_DIRS
  NAMES cusparse.h cuda.h
  HINTS ${CUSPARSE_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CUSPARSE_PATHS}
  DOC "cuSPARSE include header cusparse.h"
)
mark_as_advanced(CUSPARSE_INCLUDE_DIRS)

# Finds the libraries
#find_library(CUDA_LIBRARIES
 # NAMES cudart
 # HINTS ${CUDNN_HINTS}
  #PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  #PATHS ${CUDNN_PATHS}
  #DOC "CUDA library"
#)
#mark_as_advanced(CUDA_LIBRARIES)
find_library(CUSPARSE_LIBRARIES
  NAMES cusparse
  HINTS ${CUSPARSE_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUSPARSE_PATHS}
  DOC "cuSPARSE library"
)
mark_as_advanced(CUSPARSE_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CUSPARSE_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'cusparse.h', install CUDA/cusparse or set CUDA_ROOT")
endif()
#if(NOT CUDA_LIBRARIES)
 #   message(STATUS "Could NOT find CUDA library, install it or set CUDA_ROOT")
#endif()
if(NOT CUSPARSE_LIBRARIES)
    message(STATUS "Could NOT find cusparse library, install it or set CUDA_ROOT")
endif()

# Determines whether or not cuSPARSE was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuSPARSE DEFAULT_MSG CUSPARSE_INCLUDE_DIRS CUDA_LIBRARIES CUSPARSE_LIBRARIES)

# ===============================================================

set(CUDA_HINTS
   /home/a100/cu111
)
set(CUDAPATHS
  /home/a100/cu111
)

# Finds the include directories
find_path(CUDA_INCLUDE_DIRS
  NAMES cuda_runtime.h
  HINTS ${CUDA_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CUDA_PATHS}
  DOC "cuDA include header cuda_runtime.h"
)
mark_as_advanced(CUDA_INCLUDE_DIRS)

# Finds the libraries
find_library(CUDA_LIBRARIES
  NAMES cudart
  HINTS ${CUDA_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUDA_PATHS}
  DOC "CUDA library"
)
mark_as_advanced(CUDA_LIBRARIES)

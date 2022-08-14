cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(Mithril)
# setup compiler

set (CMAKE_C_COMPILER "mpicc")
set (CMAKE_CXX_COMPILER "mpicxx")
set (CMAKE_CUDA_COMPILER "/home/a100/cu111/bin/nvcc")
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)     
set(BOOST_ROOT /home/a100/boost-install)
set(Boost_NO_SYSTEM_PATHS ON)
# find necessary dependent packages 
find_package(cuDA REQUIRED)
find_package(cuBLAS REQUIRED)
find_package(cuDNN REQUIRED)
find_package(cuSPARSE REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
#find_package(Numa REQUIRED)
find_package(Nccl REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# cxx flags
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -lprofiler -Wl,--as-needed") # used for profiling, please delete it TODO
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgcc_s -lc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D PRINT_DEBUG_MSG") # delete this line to prevent printing debug msg
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
# include directories
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUBLAS_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})
include_directories(${CUSPARSE_INCLUDE_DIRS})
include_directories(${NCCL_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_DIRS} )
include_directories(${MPI_C_INCLUDE_DIRS} )
include_directories(${MPI_CXX_INCLUDE_DIRS} )
include_directories("${PROJECT_SOURCE_DIR}/context/include")
include_directories("${PROJECT_SOURCE_DIR}/core/include")

# finding all cpp files
file(GLOB_RECURSE CONTEXT_SRCS
    context/src/*.cc
)
add_library(context STATIC ${CONTEXT_SRCS})


file(GLOB CORE_SRCS
    #core/src/parallel/*.cc
    core/src/*.cc
)
add_library(core SHARED ${CORE_SRCS})
file(GLOB PARA_SRCS
     core/src/parallel/*.cc
     core/src/parallel/*.cu
)
add_library(parallel STATIC ${PARA_SRCS})
file(GLOB CUDA_SRCS
    core/src/cuda/*.cc
    core/src/cuda/*.cu
)
add_library(cudahelp STATIC ${CUDA_SRCS})
# dependencies
set (dependencies "")
set (dependencies ${dependencies} core)
set (dependencies ${dependencies} context)
set (dependencies ${dependencies} parallel)
set (dependencies ${dependencies} cudahelp)
set (dependencies ${dependencies} ${MPI_LIBRARIES})
#set (dependencies ${dependencies} ${NUMA_LIBRARY})
set (dependencies ${dependencies} ${Boost_LIBRARIES})
set (dependencies ${dependencies} ${CUDA_LIBRARIES})
set (dependencies ${dependencies} ${CUBLAS_LIBRARIES})
set (dependencies ${dependencies} ${CUDNN_LIBRARIES})
set (dependencies ${dependencies} ${CUSPARSE_LIBRARIES})
set (dependencies ${dependencies} ${NCCL_LIBRARY})


add_subdirectory(applications)
add_subdirectory(tests)
add_subdirectory(toolkits)


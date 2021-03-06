CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


MESSAGE("===================================================================")
MESSAGE("Specifying CXX compiler and flags...\n")
MESSAGE("NOTE: clang++ is default C++ compiler. To use other compiler, ")
MESSAGE("please define the environment variable CXX\n")

IF (NOT DEFINED ENV{CXX})
    SET(CMAKE_CXX_COMPILER "clang++")
    MESSAGE("CXX compiler: clang++")
ELSE()
    SET(CMAKE_CXX_COMPILER $ENV{CXX})
    MESSAGE("CXX compiler: " ${CMAKE_CXX_COMPILER})
ENDIF()


SET(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "")
SET(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")



MESSAGE("\n===================================================================")
MESSAGE("Specifying C compiler and flags...\n")
MESSAGE("NOTE: clang is default C compiler. To use other compiler, ")
MESSAGE("please define the environment variable CXX\n")

IF (NOT DEFINED ENV{CC})
    SET(CMAKE_C_COMPILER "clang")
    MESSAGE("C compiler: clang")
ELSE()
    SET(CMAKE_C_COMPILER $ENV{CC})
    MESSAGE("C compiler: " ${CMAKE_C_COMPILER})
ENDIF()

SET(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")
SET(CMAKE_C_FLAGS_DEBUG "-g" CACHE STRING "")



MESSAGE("\n===================================================================")
MESSAGE("Specifying other variables...")
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

IF(NOT DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING "")
ENDIF()

MESSAGE("-- Build type: " ${CMAKE_BUILD_TYPE})

SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

MESSAGE("\n===================================================================")
MESSAGE("Starting generator...")
PROJECT(AmgXtest)



MESSAGE("\n===================================================================")
MESSAGE("Trying to find AmgX...")
MESSAGE("\nNOTE: If it fails to find AmgX, please set the environment variable")
MESSAGE("AMGX_DIR in order to notify CMake the location of AmgX\n")


MESSAGE("-- Finding amgx_c.h")
FIND_PATH(AMGX_INCLUDE_DIRS 
          NAMES amgx_c.h  
          PATHS ${AMGX_DIR} $ENV{AMGX_DIR} 
          PATH_SUFFIXES "include")

IF(${AMGX_INCLUDE_DIRS} STREQUAL "AMGX_INCLUDE_DIRS-NOTFOUND")
    MESSAGE(FATAL_ERROR "Could not find amgx_c.h!!!")
ELSE()
    MESSAGE("-- Finding amgx_c.h - found in " ${AMGX_INCLUDE_DIRS})
ENDIF()


MESSAGE("-- Finding amgx_capi.h")
FIND_FILE(AMGX_CAPI_FOUND 
          NAMES amgx_capi.h 
          PATHS ${AMGX_INCLUDE_DIRS} 
          NO_DEFAULT_PATH)

IF(${AMGX_CAPI_FOUND} STREQUAL "AMGX_CAPI_FOUND-NOTFOUND")
    MESSAGE(FATAL_ERROR "amgx_capi.h was not found in: " ${AMGX_INCLUDE_DIRS})
ELSE()
    MESSAGE("-- Finding amgx_capi.h - found")
ENDIF()


MESSAGE("-- Finding amgx_config.h")
FIND_FILE(AMGX_CONFIG_FOUND 
          NAMES amgx_config.h 
          PATHS ${AMGX_INCLUDE_DIRS} 
          NO_DEFAULT_PATH)

IF(${AMGX_CONFIG_FOUND} STREQUAL "AMGX_CONFIG_FOUND-NOTFOUND")
    MESSAGE(FATAL_ERROR "amgx_config.h was not found in: " ${AMGX_INCLUDE_DIRS})
ELSE()
    MESSAGE("-- Finding amgx_config.h - found")
ENDIF()


MESSAGE("-- Finding libamgxsh.so")
FIND_LIBRARY(AMGX_LIBRARIES
             NAMES amgxsh
             PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
             PATH_SUFFIXES "lib")

IF(${AMGX_LIBRARIES} STREQUAL "AMGX_LIBRARIES-NOTFOUND")
    MESSAGE(FATAL_ERROR "libamgxsh.so was not found")
ELSE()
    MESSAGE("-- Finding libamgxsh.so - found: " ${AMGX_LIBRARIES})
ENDIF()



MESSAGE("\n===================================================================")
MESSAGE("Trying to find Boost...\n")
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options timer system mpi)

IF(${Boost_FOUND})
    MESSAGE("\tBoost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
    MESSAGE("\tBoost_LIBRARY_DIRS: " ${Boost_LIBRARY_DIRS})
    MESSAGE("\tBoost_LIBRARIES: " ${Boost_LIBRARIES})
ENDIF()



MESSAGE("\n===================================================================")
MESSAGE("Trying to find CUDA 6.5...")
MESSAGE("\nNOTE: If you have several different versions of CUDA, and if CMake ")
MESSAGE("can not find CUDA 6.5, please define CUDA_TOOLKIT_ROOT_DIR manually ")
MESSAGE("when calling CMake. For example:") 
MESSAGE("\n\t$ cmake . -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda65\n")

FIND_PACKAGE(CUDA 6.5 EXACT REQUIRED)

IF(${CUDA_FOUND})
    SET(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    MESSAGE("\tCUDA_TOOLKIT_ROOT_DIR: " ${CUDA_TOOLKIT_ROOT_DIR})
    MESSAGE("\tCUDA_VERSION: " ${CUDA_VERSION_STRING})
    MESSAGE("\tCUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
    MESSAGE("\tCUDA_LIBRARY_DIRS: " ${CUDA_LIBRARY_DIRS})
    MESSAGE("\tCUDA_LIBRARIES: " ${CUDA_LIBRARIES})
ENDIF()



MESSAGE("\n===================================================================")
MESSAGE("Trying to find MPI")

FIND_PACKAGE(MPI REQUIRED)

IF(${MPI_CXX_FOUND})
    MESSAGE("\tFound MPI: " ${MPI_CXX_FOUND})
    MESSAGE("\tMPI_CXX_COMPILER: " ${MPI_CXX_COMPILER})
    MESSAGE("\tMPI_CXX_COMPILE_FLAGS: " ${MPI_CXX_COMPILE_FLAGS})
    MESSAGE("\tMPI_CXX_INCLUDE_PATH: " ${MPI_CXX_INCLUDE_PATH})
    MESSAGE("\tMPI_CXX_LINK_FLAGS: " ${MPI_CXX_LINK_FLAGS})
    MESSAGE("\tMPI_CXX_LIBRARIES: " ${MPI_CXX_LIBRARIES})
    #SET (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} 
    #${MPI_CXX_COMPILE_FLAGS}")
                                  #SET (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} 
                                  #${MPI_CXX_LINK_FLAGS}")
ENDIF()



MESSAGE("\n===================================================================")
MESSAGE("Setting the information for target and source files...")

SET(SRC ${CMAKE_SOURCE_DIR}/src)
SET(SOURCE1 ${SRC}/sGPU.cpp ${SRC}/miscsFuncs.cpp ${SRC}/SpMt.cpp)
SET(SOURCE2 ${SRC}/mGPU_MPI.cpp ${SRC}/miscsFuncs.cpp ${SRC}/SpMt.cpp)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} 
                    ${AMGX_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${MPI_CXX_INCLUDE_PATH})

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}
                 ${CUDA_LIBRARY_DIRS}
                 ${AMGX_LIBRARY_DIRS})


ADD_EXECUTABLE(sGPU ${SOURCE1})
ADD_EXECUTABLE(mGPU_MPI ${SOURCE2})

TARGET_LINK_LIBRARIES(sGPU 
                      ${Boost_LIBRARIES}
                      ${AMGX_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${MPI_CXX_LIBRARIES})
TARGET_LINK_LIBRARIES(mGPU_MPI 
                      ${Boost_LIBRARIES}
                      ${AMGX_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${MPI_CXX_LIBRARIES})

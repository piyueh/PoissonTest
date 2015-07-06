
# Check the version of cmake
cmake_minimum_required(VERSION 2.8)

# Set the paths of Trilinos packages 
FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos
${TRILINOS_PATH})

# Test whether the cmake found Trilinos
IF(Trilinos_FOUND)
	MESSAGE("\nFound Trilinos! Here are the details: ")
	MESSAGE("\tTrilinos_DIR = ${Trilinos_DIR}")
	MESSAGE("\tTrilinos_VERSION = ${Trilinos_VERSION}")
	MESSAGE("\tTrilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
	MESSAGE("\tTrilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
	MESSAGE("\tTrilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
	MESSAGE("\tTrilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
	MESSAGE("\tTrilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
	MESSAGE("\tTrilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
	MESSAGE("\tTrilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
	MESSAGE("\tTrilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
	MESSAGE("\tTrilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
	MESSAGE("\tTrilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
	MESSAGE("\tTrilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
	MESSAGE("\tTrilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
	MESSAGE("\tTrilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
	MESSAGE("\tTrilinos_LINKER = ${Trilinos_LINKER}")
	MESSAGE("\tTrilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
	MESSAGE("\tTrilinos_AR = ${Trilinos_AR}")
	MESSAGE("End of Trilinos details\n")
ELSE()
	MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()


IF(NOT CMAKE_CXX_COMPILER)
	MESSAGE("Setting CXX compiler.")
	SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
ENDIF()


IF(NOT CMAKE_C_COMPILER)
	MESSAGE("Setting C compiler.")
	SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

# EDIT HERE! The name of this project
project(TriPoisson)


# EDIT HERE! Source files
set(SOURCES1 poisson.cpp)
set(SOURCES2 poissonSOR.cpp)
set(SOURCES3 poissonDirect.cpp)

# EDIT HERE! Headers, if any
# set(HRADERS )

# EDIT HERE! Header include path
include_directories(".")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

# EDIT HERE! Library include path
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# EDIT HERE! Executable file and corresponding source files
add_executable(TriPoissonCPU ${SOURCES1})
add_executable(TriPoissonGPU ${SOURCES1})
add_executable(TriPoissonSOR ${SOURCES2})
add_executable(TriPoissonDirect ${SOURCES3})

# EDIT HERE! I'm not sure what this command is doing...
target_link_libraries(TriPoissonCPU ${Trilinos_LIBRARIES} 
								 ${Trilinos_TPL_LIBRARIES})
target_link_libraries(TriPoissonGPU ${Trilinos_LIBRARIES} 
								 ${Trilinos_TPL_LIBRARIES})
target_link_libraries(TriPoissonSOR ${Trilinos_LIBRARIES} 
									${Trilinos_TPL_LIBRARIES})
target_link_libraries(TriPoissonDirect ${Trilinos_LIBRARIES} 
									   ${Trilinos_TPL_LIBRARIES})

# EDIT HERE!
target_compile_definitions(TriPoissonCPU PRIVATE 
	"NODETYPE=Kokkos::Compat::KokkosSerialWrapperNode")
target_compile_definitions(TriPoissonGPU PRIVATE 
	"NODETYPE=Kokkos::Compat::KokkosCudaWrapperNode")

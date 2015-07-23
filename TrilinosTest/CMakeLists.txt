
# Check the version of cmake
cmake_minimum_required(VERSION 2.8)

# Set the paths of Trilinos and its subpackages 
FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

FIND_PACKAGE(Kokkos PATHS ${TRILINOS_PATH}/lib/cmake/Kokkos ${TRILINOS_PATH})
FIND_PACKAGE(Tpetra PATHS ${TRILINOS_PATH}/lib/cmake/Tpetra ${TRILINOS_PATH})
FIND_PACKAGE(Teuchos PATHS ${TRILINOS_PATH}/lib/cmake/Teuchos ${TRILINOS_PATH})
FIND_PACKAGE(Belos PATHS ${TRILINOS_PATH}/lib/cmake/Belos ${TRILINOS_PATH})
FIND_PACKAGE(Ifpack2 PATHS ${TRILINOS_PATH}/lib/cmake/Ifpack2 ${TRILINOS_PATH})

# Test whether the cmake found necessary subpackages
IF(Kokkos_FOUND)
    MESSAGE("\nFound Kokkos!")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Kokkos!")
ENDIF()

IF(Tpetra_FOUND)
    MESSAGE("\nFound Tpetra!")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Tpetra!")
ENDIF()

IF(Teuchos_FOUND)
    MESSAGE("\nFound Teuchos!")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Teuchos!")
ENDIF()

IF(Belos_FOUND)
    MESSAGE("\nFound Belos!")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Belos!")
ENDIF()

IF(Ifpack2_FOUND)
    MESSAGE("\nFound Ifpack2!")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Ifpack2!")
ENDIF()


# set compilers
IF(NOT CMAKE_CXX_COMPILER)
    MESSAGE("Setting CXX compiler.")
    SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
ENDIF()


IF(NOT CMAKE_C_COMPILER)
    MESSAGE("Setting C compiler.")
    SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()


# The name of this project
project(TriPoisson)


# Source files
set(SOURCES1 src/poisson.cpp)
set(SOURCES2 src/MVM.cpp)


# Header include path
include_directories(".")
include_directories(${Kokkos_TPL_INCLUDE_DIRS} 
                    ${Tpetra_TPL_INCLUDE_DIRS} 
                    ${Teuchos_TPL_INCLUDE_DIRS} 
                    ${Belos_TPL_INCLUDE_DIRS} 
                    ${Ifpack2_TPL_INCLUDE_DIRS} )
include_directories(${Kokkos_INCLUDE_DIRS} 
                    ${Tpetra_INCLUDE_DIRS} 
                    ${Teuchos_INCLUDE_DIRS} 
                    ${Belos_INCLUDE_DIRS} 
                    ${Ifpack2_INCLUDE_DIRS} )

# Library include path
link_directories(${Kokkos_LIBRARY_DIRS} 
                 ${Tpetra_LIBRARY_DIRS} 
                 ${Teuchos_LIBRARY_DIRS} 
                 ${Belos_LIBRARY_DIRS} 
                 ${Ifpack2_LIBRARY_DIRS} )
link_directories(${Kokkos_TPL_LIBRARY_DIRS} 
                 ${Tpetra_TPL_LIBRARY_DIRS} 
                 ${Teuchos_TPL_LIBRARY_DIRS} 
                 ${Belos_TPL_LIBRARY_DIRS} 
                 ${Ifpack2_TPL_LIBRARY_DIRS} )


# Executable file and corresponding source files
add_executable(build/TriPoisson_MpiThreads ${SOURCES1})
add_executable(build/TriPoisson_MpiCuda ${SOURCES1})
add_executable(build/MVM_MpiThreads ${SOURCES2})
add_executable(build/MVM_MpiCuda ${SOURCES2})

# I'm not sure what this command is doing...
target_link_libraries(build/TriPoisson_MpiThreads ${Kokkos_LIBRARIES} 
                                                  ${Kokkos_TPL_LIBRARIES}
                                                  ${Tpetra_LIBRARIES}
                                                  ${Tpetra_TPL_LIBRARIES}
                                                  ${Teuchos_LIBRARIES}
                                                  ${Teuchos_TPL_LIBRARIES}
                                                  ${Belos_LIBRARIES}
                                                  ${Belos_TPL_LIBRARIES}
                                                  ${Ifpack2_LIBRARIES}
                                                  ${Ifpack2_TPL_LIBRARIES})
target_link_libraries(build/TriPoisson_MpiCuda ${Kokkos_LIBRARIES} 
                                               ${Kokkos_TPL_LIBRARIES}
                                               ${Tpetra_LIBRARIES}
                                               ${Tpetra_TPL_LIBRARIES}
                                               ${Teuchos_LIBRARIES}
                                               ${Teuchos_TPL_LIBRARIES}
                                               ${Belos_LIBRARIES}
                                               ${Belos_TPL_LIBRARIES}
                                               ${Ifpack2_LIBRARIES}
                                               ${Ifpack2_TPL_LIBRARIES})
target_link_libraries(build/MVM_MpiThreads ${Kokkos_LIBRARIES} 
                                           ${Kokkos_TPL_LIBRARIES}
                                           ${Tpetra_LIBRARIES}
                                           ${Tpetra_TPL_LIBRARIES}
                                           ${Teuchos_LIBRARIES}
                                           ${Teuchos_TPL_LIBRARIES})
target_link_libraries(build/MVM_MpiCuda ${Kokkos_LIBRARIES} 
                                        ${Kokkos_TPL_LIBRARIES}
                                        ${Tpetra_LIBRARIES}
                                        ${Tpetra_TPL_LIBRARIES}
                                        ${Teuchos_LIBRARIES}
                                        ${Teuchos_TPL_LIBRARIES})

# target-specific macro variables
target_compile_definitions(build/TriPoisson_MpiThreads PRIVATE 
    "NODETYPE=Kokkos::Compat::KokkosThreadsWrapperNode")
target_compile_definitions(build/TriPoisson_MpiCuda PRIVATE 
    "NODETYPE=Kokkos::Compat::KokkosCudaWrapperNode")
target_compile_definitions(build/MVM_MpiThreads PRIVATE 
    "NODETYPE=Kokkos::Compat::KokkosThreadsWrapperNode")
target_compile_definitions(build/MVM_MpiCuda PRIVATE 
    "NODETYPE=Kokkos::Compat::KokkosCudaWrapperNode")

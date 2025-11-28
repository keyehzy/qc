cmake_minimum_required(VERSION 3.15)

# Set compilers
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

project(QC CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable sanitizers
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# AddressSanitizer and UndefinedBehaviorSanitizer (conditional)
if(ENABLE_SANITIZERS)
    message(STATUS "Sanitizers enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address,undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GSL paths
set(GSL_ROOT_DIR "/opt/homebrew/Cellar/gsl/2.8")
set(GSL_INCLUDE_DIR "${GSL_ROOT_DIR}/include")
set(GSL_LIBRARY_DIR "${GSL_ROOT_DIR}/lib")

# OpenBLAS paths
set(OPENBLAS_ROOT_DIR "/opt/homebrew/opt/openblas")
set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_ROOT_DIR}/include")
set(OPENBLAS_LIBRARY_DIR "${OPENBLAS_ROOT_DIR}/lib")

# Eigen paths
set(EIGEN_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include")

# OpenMP paths (libomp)
set(LIBOMP_ROOT_DIR "/opt/homebrew/Cellar/libomp/21.1.2")
set(LIBOMP_INCLUDE_DIR "${LIBOMP_ROOT_DIR}/include")
set(LIBOMP_LIBRARY_DIR "${LIBOMP_ROOT_DIR}/lib")

# Include directories
include_directories(
    ${GSL_INCLUDE_DIR}
    ${OPENBLAS_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${LIBOMP_INCLUDE_DIR}
)

# Eigen compile definitions
add_compile_definitions(EIGEN_USE_BLAS EIGEN_USE_LAPACKE)

# Library directories
link_directories(
    ${GSL_LIBRARY_DIR}
    ${OPENBLAS_LIBRARY_DIR}
    ${LIBOMP_LIBRARY_DIR}
)

# Source files
set(SOURCES
    src/main.cpp
    src/orbital.cpp
    src/basis_set.cpp
    src/hartree_fock/restricted_hf.cpp
    src/hartree_fock/unrestricted_hf.cpp
    src/lda.cpp
    src/uniform_grid.cpp
    src/atom_centered_grid.cpp
)

# Executable
add_executable(main ${SOURCES})

# Link libraries
target_link_libraries(main
    gsl
    gslcblas
    m
    openblas
    OpenMP::OpenMP_CXX
)
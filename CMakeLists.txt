project(LeducHoldem)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "D:/ProgramFiles/Anaconda3/Lib/site-packages/torch/share/cmake")
message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
if(MSVC)
  # THPVariable_Wrap error
  find_library(TORCH_PYTHON_LIBRARY torch_python ${TORCH_INSTALL_PREFIX}/lib REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

include_directories("./cpp")
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${MKL_ROOT}/lib/intel64)

add_subdirectory(D:/ProgramFiles/pybind11-2.3.0 pybind11 EXCLUDE_FROM_ALL)
pybind11_add_module(rela cpp/pybind.cpp)
target_link_libraries(rela PRIVATE torch ${TORCH_PYTHON_LIBRARY})
# target_include_directories(rela PRIVATE ${PYTHON_INCLUDE_DIRS})

include_directories("D:/ProgramFiles/eigen-3.4.0")
add_executable(test cpp/test.cpp)
target_link_libraries(test torch ${TORCH_PYTHON_LIBRARY})
target_compile_options(test PRIVATE "/openmp")
add_executable(solver cpp/main.cpp)
target_link_libraries(solver torch ${TORCH_PYTHON_LIBRARY})
target_compile_options(solver PRIVATE "/openmp")
add_executable(recursive_eval cpp/recursive_eval.cpp)
target_link_libraries(recursive_eval torch ${TORCH_PYTHON_LIBRARY})
target_compile_options(recursive_eval PRIVATE "/openmp")

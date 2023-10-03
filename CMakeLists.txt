cmake_minimum_required(VERSION 3.27)

project(nn_simple
        VERSION 0.0.1
        LANGUAGES CXX)
        
set(CMAKE_CXX_STANDARD 23)

# set a default build type: Debug | RelWithDebInfo | Release | MinSizeRel
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Build type not specified: use Debug mode by default.")
endif()

if(CMAKE_BUILD_TYPE EQUAL "Debug")
  if(MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
  else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
  endif()
endif()

add_executable(nn
               src/main.cpp
               src/neural_net_func.cpp
               src/neural_net.cpp
               src/neural_net_file_io.cpp
               src/neural_net_print.cpp)
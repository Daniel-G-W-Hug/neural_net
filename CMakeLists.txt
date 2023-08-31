cmake_minimum_required(VERSION 3.25)

project(nn_simple
        VERSION 0.0.1
        LANGUAGES CXX)
        
set(CMAKE_CXX_STANDARD 20)

add_executable(nn src/main.cpp src/activation_func.cpp src/neural_net.cpp src/file_io.cpp)
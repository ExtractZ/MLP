cmake_minimum_required(VERSION 3.25)

project(MLXinferenceEngine VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../mlx"
    "${CMAKE_CURRENT_BINARY_DIR}/mlx"
    EXCLUDE_FROM_ALL
)

add_executable(inference_engine main.cpp)
target_link_libraries(inference_engine PRIVATE mlx)
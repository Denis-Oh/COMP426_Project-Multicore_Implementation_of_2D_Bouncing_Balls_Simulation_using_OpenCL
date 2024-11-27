cmake_minimum_required(VERSION 3.10)
project(BallSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)

# Find GLFW
if (APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

# Add executable
add_executable(BallSimulation main.cpp)

# Link libraries
if (APPLE)
    target_link_libraries(BallSimulation
        "-framework OpenGL"
        "-framework OpenCL"
        glfw
    )
else()
    target_link_libraries(BallSimulation
        OpenGL::GL
        OpenCL::OpenCL
        glfw
    )
endif()

# Copy kernel files to build directory
configure_file(${CMAKE_SOURCE_DIR}/gpu_kernel.cl ${CMAKE_BINARY_DIR}/gpu_kernel.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/cpu_kernel.cl ${CMAKE_BINARY_DIR}/cpu_kernel.cl COPYONLY)
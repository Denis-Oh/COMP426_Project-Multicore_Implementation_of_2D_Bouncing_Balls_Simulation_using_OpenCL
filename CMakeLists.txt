cmake_minimum_required(VERSION 3.10)
project(BouncingBallsSimulation)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Set up framework paths for macOS
set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

# Set GLFW paths
set(GLFW_DIR "/opt/homebrew/opt/glfw")
include_directories(${GLFW_DIR}/include)
find_library(GLFW_LIBRARY glfw PATHS ${GLFW_DIR}/lib)

# Include OpenGL
find_library(OpenGL_LIBRARY OpenGL)

# Add the executable
add_executable(BouncingBalls main.cpp)

# Include directories for macOS
target_include_directories(BouncingBalls PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${GLFW_DIR}/include
    "/System/Library/Frameworks/OpenCL.framework/Headers"
)

# Link frameworks and libraries
target_link_libraries(BouncingBalls
    "-framework OpenCL"
    ${GLFW_LIBRARY}
    ${OpenGL_LIBRARY}
)

# Copy both kernel files to build directory
configure_file(${CMAKE_SOURCE_DIR}/gpu_kernel.cl ${CMAKE_BINARY_DIR}/gpu_kernel.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/cpu_kernel.cl ${CMAKE_BINARY_DIR}/cpu_kernel.cl COPYONLY)
cmake_minimum_required(VERSION 3.10)
project(BallSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add definition to silence OpenGL deprecation warnings
add_definitions(-DGL_SILENCE_DEPRECATION)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Include directories for M1 Mac
include_directories(
    /opt/homebrew/include
    /System/Library/Frameworks/OpenCL.framework/Headers
    /System/Library/Frameworks/OpenGL.framework/Headers
    ${CMAKE_SOURCE_DIR}  # Add this line to include the source directory
)

# Link directories for M1 Mac
link_directories(/opt/homebrew/lib)

# Add executable
add_executable(BallSimulation main.cpp)

# Link frameworks and libraries for M1 Mac
target_link_libraries(BallSimulation
    "-framework OpenGL"
    "-framework OpenCL"
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
    "/opt/homebrew/lib/libglfw.3.dylib"
)

# Copy kernel files to build directory
configure_file(${CMAKE_SOURCE_DIR}/gpu_kernel.cl ${CMAKE_BINARY_DIR}/gpu_kernel.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/cpu_kernel.cl ${CMAKE_BINARY_DIR}/cpu_kernel.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/ball_def.h ${CMAKE_BINARY_DIR}/ball_def.h COPYONLY)
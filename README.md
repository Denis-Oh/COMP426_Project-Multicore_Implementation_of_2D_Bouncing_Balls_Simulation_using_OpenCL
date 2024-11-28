# COMP426_Project-Multicore_Implementation_of_2D_Bouncing_Balls_Simulation_using_OpenCL

Name: Denis Oh

Student ID: 40208580

![BouncingBallsDemo](https://github.com/user-attachments/assets/7398048c-47db-49b0-816c-f88c7a9f2b1f)

#### How to run: (from project directory)
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `./BouncingBalls`

## Introduction:
This report presents the implementation of a 2D bouncing balls simulation using OpenCL to achieve parallel processing. The project aims to leverage the unified memory architecture of the M1 chip to simulate the separation of CPU and GPU tasks while demonstrating an understanding of parallel programming principles.

## Work Division and Kernel Design
Considering the unified memory architecture of the M1 chip, the work was divided to simulate CPU and GPU tasks, even though they execute on the same physical processor. This approach allows for a clear demonstration of parallel programming concepts while accommodating the hardware limitations.

### GPU Tasks
The updateBallPositions kernel is designed to handle data-parallel computations, such as updating ball positions, applying gravity, and detecting wall collisions. Each ball is assigned to a separate work item, allowing for parallel execution.

The kernel is optimized to minimize global memory access by utilizing local memory for intermediate calculations.

### CPU Tasks
The checkBallCollisions kernel is designed to simulate task parallelism, focusing on ball-to-ball collision detection and resolution.

Collision resolution is performed using impulse-based physics calculations, ensuring realistic ball interactions.

##  Host Program and OpenCL Integration
The host program (main.cpp) is responsible for initializing the OpenCL environment, managing data transfers between the host and device, and coordinating kernel execution.

### OpenCL Initialization
The program queries the available OpenCL platforms and devices, selecting the default device for execution.

It creates an OpenCL context, command queue, and builds the kernel programs from source files.

### Memory Management
The program creates OpenCL buffer objects to store ball data. (positions, velocities, and collision statistics)

It efficiently manages memory transfers between the host and device to minimize data movement overhead.

### Kernel Execution and Synchronization
The host program sets up kernel arguments and enqueues the kernels for execution on the device.

It ensures proper synchronization between kernel executions using OpenCL events and barriers.

### Rendering and Display
The program utilizes OpenGL to render the balls on the screen, leveraging the GPU's rendering capabilities.

It uses OpenCL to efficiently transfer ball data from OpenCL buffers to OpenGL for rendering.

## Performance
Although the M1 architecture provides unified memory, the implementation still aims to optimize performance by minimizing data movement and maximizing parallel execution.

### Workgroup Size
The kernels are launched with an appropriate workgroup size to ensure efficient utilization of the GPU's compute units.

The workgroup size is chosen based on the number of balls and the device's capabilities to achieve optimal occupancy.

### Memory Access
Local memory is utilized for intermediate calculations to reduce global memory accesses thus improving performance.

## Conclusion
The implementation of the 2D bouncing balls simulation using OpenCL on the M1 architecture demonstrates the application of parallel programming principles while accommodating my hardware limitations. By simulating the separation of CPU and GPU tasks, the project showcases the benefits of parallel execution and efficient memory management.

The host program effectively coordinates the execution of kernels, managing data transfers and synchronization. The kernels are designed to maximize parallel execution and optimize memory access, resulting in improved performance.

Despite the unified memory architecture of the M1 chip, the project successfully illustrates the concepts of data parallelism and task parallelism, providing a solid foundation for understanding parallel programming techniques.




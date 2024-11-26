#include <GLFW/glfw3.h>
#include <OpenCL/opencl.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

struct Ball {
    cl_float4 position;  // x, y, radius, padding
    cl_float4 velocity;  // vx, vy, padding, padding
    cl_float4 color;     // r, g, b, padding
};

std::vector<Ball> balls;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem d_positions, d_velocities, d_colors;

// Function to check OpenCL errors
void checkError(cl_int error, const char* location) {
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL error at " << location << ": " << error << std::endl;
        exit(error);
    }
}

// Function to load and compile OpenCL kernel
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

void initOpenCL() {
    cl_int error;
    cl_device_id device;  // Declare device here to use it later
    
    // Get platform
    cl_platform_id platform;
    cl_uint num_platforms;
    error = clGetPlatformIDs(1, &platform, &num_platforms);
    checkError(error, "clGetPlatformIDs");
    
    // Get device
    cl_uint num_devices;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    checkError(error, "clGetDeviceIDs");
    
    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError(error, "clCreateContext");
    
    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    checkError(error, "clCreateCommandQueue");
    
    // Load and build program
    std::string kernel_source = loadKernel("kernel.cl");
    const char* source_ptr = kernel_source.c_str();
    size_t source_len = kernel_source.length();
    program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &error);
    checkError(error, "clCreateProgramWithSource");
    
    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // Get build log size
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        // Get build log
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        std::cerr << "OpenCL build error:\n" << log.data() << std::endl;
        exit(error);
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "updateBalls", &error);
    checkError(error, "clCreateKernel");
    
    // Get maximum work group size
    size_t max_work_group_size;
    error = clGetKernelWorkGroupInfo(kernel, device, 
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    checkError(error, "clGetKernelWorkGroupInfo");
    
    std::cout << "Maximum work group size: " << max_work_group_size << std::endl;
}

void initializeBalls(int numBalls) {
    srand(static_cast<unsigned int>(time(0)));
    balls.resize(numBalls);
    
    // Create separate vectors for each component
    std::vector<cl_float4> positions(numBalls);
    std::vector<cl_float4> velocities(numBalls);
    std::vector<cl_float4> colors(numBalls);
    
    // radius
    float radius_options[] = {50.0f, 100.0f, 150.0f};
    
    std::cout << "Initializing balls:" << std::endl;
    
    for (int i = 0; i < numBalls; ++i) {
        // Random radius (50, 100, or 150)
        float radius = radius_options[rand() % 3];
        
        // Random position (ensuring ball doesn't spawn partially outside window)
        positions[i].s[0] = radius + (rand() % (int)(1200 - 2 * radius));    // x
        positions[i].s[1] = radius + (rand() % (int)(900 - 2 * radius));    // y
        positions[i].s[2] = radius;                                         // radius
        positions[i].s[3] = 0.0f;                                          // padding
        
        // Random velocity (-1.0 to 1.0 for both x and y)
        velocities[i].s[0] = (float)(rand() % 200 - 100) / 50.0f;  // vx
        velocities[i].s[1] = (float)(rand() % 200 - 100) / 50.0f;  // vy
        velocities[i].s[2] = 0.0f;                                 // padding
        velocities[i].s[3] = 0.0f;                                 // padding
        
        // Colors (red, green, or blue)
        if (i % 3 == 0) {
            colors[i].s[0] = 1.0f; colors[i].s[1] = 0.0f; colors[i].s[2] = 0.0f;  // Red
        } else if (i % 3 == 1) {
            colors[i].s[0] = 0.0f; colors[i].s[1] = 1.0f; colors[i].s[2] = 0.0f;  // Green
        } else {
            colors[i].s[0] = 0.0f; colors[i].s[1] = 0.0f; colors[i].s[2] = 1.0f;  // Blue
        }
        colors[i].s[3] = 0.0f;  // padding
        
        // Store in balls array for CPU-side reference
        balls[i].position = positions[i];
        balls[i].velocity = velocities[i];
        balls[i].color = colors[i];
        
        std::cout << "Ball " << i << ": "
                  << "pos=(" << positions[i].s[0] << "," << positions[i].s[1] << ") "
                  << "vel=(" << velocities[i].s[0] << "," << velocities[i].s[1] << ") "
                  << "radius=" << positions[i].s[2]
                  << std::endl;
    }
    
    cl_int error;
    
    // Create buffers
    d_positions = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(cl_float4) * numBalls, nullptr, &error);
    checkError(error, "clCreateBuffer positions");
    
    d_velocities = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(cl_float4) * numBalls, nullptr, &error);
    checkError(error, "clCreateBuffer velocities");
    
    d_colors = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(cl_float4) * numBalls, nullptr, &error);
    checkError(error, "clCreateBuffer colors");
    
    // Copy data to device
    error = clEnqueueWriteBuffer(queue, d_positions, CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, positions.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueWriteBuffer positions");
    
    error = clEnqueueWriteBuffer(queue, d_velocities, CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, velocities.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueWriteBuffer velocities");
    
    error = clEnqueueWriteBuffer(queue, d_colors, CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, colors.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueWriteBuffer colors");
    
    std::cout << "Initialization complete" << std::endl;
}

void drawBalls() {
    // Read back all data
    std::vector<cl_float4> positions(balls.size());
    std::vector<cl_float4> velocities(balls.size());
    std::vector<cl_float4> colors(balls.size());
    
    cl_int error = clEnqueueReadBuffer(queue, d_positions, CL_TRUE, 0,
        sizeof(cl_float4) * balls.size(), positions.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueReadBuffer positions");
    
    error = clEnqueueReadBuffer(queue, d_velocities, CL_TRUE, 0,
        sizeof(cl_float4) * balls.size(), velocities.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueReadBuffer velocities");
    
    error = clEnqueueReadBuffer(queue, d_colors, CL_TRUE, 0,
        sizeof(cl_float4) * balls.size(), colors.data(), 0, nullptr, nullptr);
    checkError(error, "clEnqueueReadBuffer colors");

    // Print debug info
    static int frame_count = 0;
    if (frame_count++ % 60 == 0) {  // Print every 60 frames
        std::cout << "\nBall States:" << std::endl;
        for (size_t i = 0; i < balls.size(); i++) {
            std::cout << "Ball " << i << ": "
                      << "pos=(" << positions[i].s[0] << "," << positions[i].s[1] << ") "
                      << "vel=(" << velocities[i].s[0] << "," << velocities[i].s[1] << ")"
                      << std::endl;
        }
        std::cout << std::endl;
    }

    // Draw balls
    for (size_t i = 0; i < balls.size(); i++) {
        glColor3f(colors[i].s[0], colors[i].s[1], colors[i].s[2]);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(positions[i].s[0], positions[i].s[1]);
        
        for (int j = 0; j <= 360; ++j) {
            float angle = j * 3.14159f / 180.0f;
            glVertex2f(
                positions[i].s[0] + cos(angle) * positions[i].s[2],
                positions[i].s[1] + sin(angle) * positions[i].s[2]
            );
        }
        glEnd();
    }
}

void cleanupOpenCL() {
    clReleaseMemObject(d_positions);
    clReleaseMemObject(d_velocities);
    clReleaseMemObject(d_colors);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, "OpenCL 2D Bouncing Balls", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize OpenGL
    glViewport(0, 0, 1200, 900);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1200, 0, 900, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    try {
        // Initialize OpenCL
        initOpenCL();
        initializeBalls(5);
        
        // Set kernel arguments
        cl_int error;
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_positions);
        checkError(error, "clSetKernelArg 0");
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_velocities);
        checkError(error, "clSetKernelArg 1");
        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_colors);
        checkError(error, "clSetKernelArg 2");
        
        int num_balls = balls.size();
        error = clSetKernelArg(kernel, 3, sizeof(int), &num_balls);
        checkError(error, "clSetKernelArg 3");
        
        float gravity = -0.03f;
        error = clSetKernelArg(kernel, 4, sizeof(float), &gravity);
        checkError(error, "clSetKernelArg 4");
        
        float window_width = 1200.0f;
        error = clSetKernelArg(kernel, 5, sizeof(float), &window_width);
        checkError(error, "clSetKernelArg 5");
        
        float window_height = 900.0f;
        error = clSetKernelArg(kernel, 6, sizeof(float), &window_height);
        checkError(error, "clSetKernelArg 6");
        
        // Set up work sizes for parallel execution
        const int THREADS_PER_BALL = 32;
        const int LOCAL_SIZE = 256;
        size_t global_work_size = ((balls.size() * THREADS_PER_BALL + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
        size_t local_work_size = LOCAL_SIZE;

        std::cout << "Global work size: " << global_work_size << std::endl;
        std::cout << "Local work size: " << local_work_size << std::endl;
        
        // Performance measurement variables
        float total_frame_time = 0.0f;
        int frame_count = 0;
        
        while (!glfwWindowShouldClose(window)) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Launch kernel with parallel configuration
            error = clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,              // One dimension
                nullptr,        // Global work offset
                &global_work_size,
                &local_work_size,
                0, nullptr, nullptr
            );
            checkError(error, "clEnqueueNDRangeKernel");
            
            error = clFinish(queue);
            checkError(error, "clFinish");
            
            // Render
            glClear(GL_COLOR_BUFFER_BIT);
            drawBalls();
            glfwSwapBuffers(window);
            glfwPollEvents();
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // Frame timing and performance metrics
            total_frame_time += duration.count();
            frame_count++;
            
            if (frame_count % 100 == 0) {
                float avg_frame_time = total_frame_time / frame_count;
                float fps = 1000.0f / avg_frame_time;
                std::cout << "Average frame time: " << avg_frame_time << "ms (" << fps << " FPS)" << std::endl;
                total_frame_time = 0;
                frame_count = 0;
            }
            
            // if (duration.count() < 33) {
            //     std::this_thread::sleep_for(std::chrono::milliseconds(33 - duration.count()));
            // }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cleanupOpenCL();
        glfwTerminate();
        return -1;
    }
    
    cleanupOpenCL();
    glfwTerminate();
    return 0;
}
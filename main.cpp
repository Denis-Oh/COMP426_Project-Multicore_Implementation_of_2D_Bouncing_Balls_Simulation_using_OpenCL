#include <OpenCL/cl.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include "ball_def.h"

// Constants
const int NUM_BALLS = 100;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const float MIN_RADIUS = 5.0f;
const float MAX_RADIUS = 15.0f;
const float MAX_INITIAL_VELOCITY = 200.0f;

// OpenCL variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel gpuKernel, cpuKernel;
cl_mem ballBuffer, vertexBuffer, statsBuffer;

// GLFW window
GLFWwindow* window = nullptr;

// Utility function to read kernel source
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    return std::string(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>());
}

void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error during operation " << operation << ": " << error << std::endl;
        exit(1);
    }
}

// Initialize OpenCL
void initOpenCL() {
    cl_int error;

    // Get platform
    cl_uint numPlatforms;
    error = clGetPlatformIDs(1, &platform, &numPlatforms);
    checkError(error, "getting platform ID");

    // Get device
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    checkError(error, "getting device");

    // Create context
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };
    context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
    checkError(error, "creating context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    checkError(error, "creating command queue");

    // Read ball_def.h content
    std::string headerContent = readFile("ball_def.h");
    
    // Read kernel sources
    std::string gpuSource = readFile("gpu_kernel.cl");
    std::string cpuSource = readFile("cpu_kernel.cl");
    
    // Combine header with each kernel
    std::string combinedGPUSource = headerContent + "\n" + gpuSource;
    std::string combinedCPUSource = headerContent + "\n" + cpuSource;
    
    // Create and build programs separately
    const char* gpuSrc = combinedGPUSource.c_str();
    size_t gpuLen = combinedGPUSource.length();
    cl_program gpuProgram = clCreateProgramWithSource(context, 1, &gpuSrc, &gpuLen, &error);
    checkError(error, "creating GPU program");
    
    error = clBuildProgram(gpuProgram, 0, nullptr, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(gpuProgram, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << "GPU Build error: " << buffer << std::endl;
        exit(1);
    }
    
    const char* cpuSrc = combinedCPUSource.c_str();
    size_t cpuLen = combinedCPUSource.length();
    cl_program cpuProgram = clCreateProgramWithSource(context, 1, &cpuSrc, &cpuLen, &error);
    checkError(error, "creating CPU program");
    
    error = clBuildProgram(cpuProgram, 0, nullptr, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(cpuProgram, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << "CPU Build error: " << buffer << std::endl;
        exit(1);
    }

    // Create kernels
    gpuKernel = clCreateKernel(gpuProgram, "updateBallPositions", &error);
    checkError(error, "creating GPU kernel");
    cpuKernel = clCreateKernel(cpuProgram, "checkBallCollisions", &error);
    checkError(error, "creating CPU kernel");

    // Create buffers
    ballBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Ball) * NUM_BALLS, nullptr, &error);
    checkError(error, "creating ball buffer");
    vertexBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * NUM_BALLS, nullptr, &error);
    checkError(error, "creating vertex buffer");
    statsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
    checkError(error, "creating stats buffer");
}

// Initialize GLFW and OpenGL
void initGraphics() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(1);
    }
    
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Bouncing Balls", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync
}

// Initialize balls with random positions and velocities
void initBalls() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posX(MAX_RADIUS, WINDOW_WIDTH - MAX_RADIUS);
    std::uniform_real_distribution<float> posY(MAX_RADIUS, WINDOW_HEIGHT - MAX_RADIUS);
    std::uniform_real_distribution<float> vel(-MAX_INITIAL_VELOCITY, MAX_INITIAL_VELOCITY);
    std::uniform_real_distribution<float> radius(MIN_RADIUS, MAX_RADIUS);
    
    std::vector<Ball> balls(NUM_BALLS);
    for (auto& ball : balls) {
        ball.position = {posX(gen), posY(gen)};  
        ball.velocity = {vel(gen), vel(gen)};    
        ball.radius = radius(gen);
    }
    
    cl_int error = clEnqueueWriteBuffer(queue, ballBuffer, CL_TRUE, 0, 
                                       sizeof(Ball) * NUM_BALLS, balls.data(), 
                                       0, nullptr, nullptr);
    checkError(error, "writing initial ball data");
}

// Render function
void render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    
    // Read ball data back
    std::vector<Ball> balls(NUM_BALLS);
    cl_int error = clEnqueueReadBuffer(queue, ballBuffer, CL_TRUE, 0,
                                      sizeof(Ball) * NUM_BALLS, balls.data(),
                                      0, nullptr, nullptr);
    checkError(error, "reading ball data for rendering");
    
    // Draw balls
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (const auto& ball : balls) {
        glVertex2f(ball.position.s[0], ball.position.s[1]);
    }
    glEnd();
    
    glfwSwapBuffers(window);
}

void cleanup() {
    clReleaseMemObject(ballBuffer);
    clReleaseMemObject(vertexBuffer);
    clReleaseMemObject(statsBuffer);
    clReleaseKernel(gpuKernel);
    clReleaseKernel(cpuKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    // Initialize OpenCL and Graphics
    initOpenCL();
    initGraphics();
    initBalls();
    
    // Set up OpenGL viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    
    // Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // Set kernel arguments and run GPU kernel
        FLOAT2 boundaries = {WINDOW_WIDTH, WINDOW_HEIGHT};
        cl_int error;

        error = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &ballBuffer);
        error |= clSetKernelArg(gpuKernel, 1, sizeof(float), &deltaTime);
        error |= clSetKernelArg(gpuKernel, 2, sizeof(FLOAT2), &boundaries);
        error |= clSetKernelArg(gpuKernel, 3, sizeof(int), &NUM_BALLS);
        checkError(error, "setting GPU kernel arguments");
        
        size_t globalSize = NUM_BALLS;
        error = clEnqueueNDRangeKernel(queue, gpuKernel, 1, nullptr, &globalSize, 
                                      nullptr, 0, nullptr, nullptr);
        checkError(error, "enqueueing GPU kernel");
        
        // Set kernel arguments and run CPU kernel (now running on the same device)
        error = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &ballBuffer);
        error |= clSetKernelArg(cpuKernel, 1, sizeof(int), &NUM_BALLS);
        error |= clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &statsBuffer);
        checkError(error, "setting CPU kernel arguments");
        
        error = clEnqueueNDRangeKernel(queue, cpuKernel, 1, nullptr, &globalSize, 
                                      nullptr, 0, nullptr, nullptr);
        checkError(error, "enqueueing CPU kernel");
        
        // Render
        render();
        
        // Poll events
        glfwPollEvents();
    }
    
    // Cleanup
    clReleaseMemObject(ballBuffer);
    clReleaseMemObject(vertexBuffer);
    clReleaseMemObject(statsBuffer);
    clReleaseKernel(gpuKernel);
    clReleaseKernel(cpuKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
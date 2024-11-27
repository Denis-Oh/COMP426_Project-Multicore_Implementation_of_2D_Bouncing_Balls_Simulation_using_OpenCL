#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>

// Ball structure matching OpenCL kernels
struct Ball {
    cl_float2 position;
    cl_float2 velocity;
    float radius;
};

// Constants
const int NUM_BALLS = 100;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const float MIN_RADIUS = 5.0f;
const float MAX_RADIUS = 15.0f;
const float MAX_INITIAL_VELOCITY = 200.0f;

// OpenCL variables
cl::Context context;
cl::CommandQueue gpuQueue;
cl::CommandQueue cpuQueue;
cl::Program program;
cl::Kernel gpuKernel;
cl::Kernel cpuKernel;
cl::Buffer ballBuffer;
cl::Buffer vertexBuffer;
cl::Buffer statsBuffer;

// GLFW window
GLFWwindow* window = nullptr;

// Utility function to read kernel source
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>());
}

// Initialize OpenCL
void initOpenCL() {
    try {
        // Get platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        // Find a platform with both CPU and GPU devices
        cl::Platform platform;
        for (const auto& p : platforms) {
            std::vector<cl::Device> devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            
            bool hasCPU = false, hasGPU = false;
            for (const auto& device : devices) {
                if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) hasCPU = true;
                if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) hasGPU = true;
            }
            
            if (hasCPU && hasGPU) {
                platform = p;
                break;
            }
        }
        
        // Create context
        context = cl::Context(CL_DEVICE_TYPE_ALL, nullptr, nullptr, nullptr);
        
        // Get devices
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device cpuDevice, gpuDevice;
        
        for (const auto& device : devices) {
            if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
                cpuDevice = device;
            else if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
                gpuDevice = device;
        }
        
        // Create command queues
        gpuQueue = cl::CommandQueue(context, gpuDevice);
        cpuQueue = cl::CommandQueue(context, cpuDevice);
        
        // Build program
        std::string gpuSource = readFile("gpu_kernel.cl");
        std::string cpuSource = readFile("cpu_kernel.cl");
        program = cl::Program(context, gpuSource + cpuSource);
        program.build(devices);
        
        // Create kernels
        gpuKernel = cl::Kernel(program, "updateBallPositions");
        cpuKernel = cl::Kernel(program, "checkBallCollisions");
        
        // Create buffers
        ballBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Ball) * NUM_BALLS);
        vertexBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * NUM_BALLS);
        statsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int));
        
    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        exit(1);
    }
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
    
    gpuQueue.enqueueWriteBuffer(ballBuffer, CL_TRUE, 0, sizeof(Ball) * NUM_BALLS, balls.data());
}

// Main rendering function
void render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    
    // Read ball data back from OpenCL buffer
    std::vector<Ball> balls(NUM_BALLS);
    gpuQueue.enqueueReadBuffer(ballBuffer, CL_TRUE, 0, sizeof(Ball) * NUM_BALLS, balls.data());
    
    // Draw balls
    glBegin(GL_POINTS);
    for (const auto& ball : balls) {
        glVertex2f(ball.position.x, ball.position.y);
    }
    glEnd();
    
    glfwSwapBuffers(window);
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
        
        // Update ball positions (GPU)
        gpuKernel.setArg(0, ballBuffer);
        gpuKernel.setArg(1, deltaTime);
        gpuKernel.setArg(2, cl_float2{WINDOW_WIDTH, WINDOW_HEIGHT});
        gpuKernel.setArg(3, NUM_BALLS);
        gpuQueue.enqueueNDRangeKernel(gpuKernel, cl::NullRange, cl::NDRange(NUM_BALLS));
        
        // Check ball collisions (CPU)
        cpuKernel.setArg(0, ballBuffer);
        cpuKernel.setArg(1, NUM_BALLS);
        cpuKernel.setArg(2, statsBuffer);
        cpuQueue.enqueueNDRangeKernel(cpuKernel, cl::NullRange, cl::NDRange(NUM_BALLS));
        
        // Render
        render();
        
        // Poll events
        glfwPollEvents();
    }
    
    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
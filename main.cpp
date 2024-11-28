#include <OpenCL/cl.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include "ball_def.h"

// Constants
const int NUM_BALLS = 30;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const float MIN_RADIUS = 15.0f;    
const float MAX_RADIUS = 25.0f;   
const float MAX_INITIAL_VELOCITY = 500.0f;

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

    // Print platform info
    char platformName[128];
    error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), 
                             platformName, nullptr);
    checkError(error, "getting platform info");
    std::cout << "OpenCL Platform: " << platformName << std::endl;

    // For M1, we want the default device
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    checkError(error, "getting device");

    // Print device info
    char deviceName[128];
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), 
                           deviceName, nullptr);
    checkError(error, "getting device info");
    std::cout << "OpenCL Device: " << deviceName << std::endl;

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

    // Required for macOS to work correctly
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);  // Add this for Retina displays
    glfwWindowHint(GLFW_SAMPLES, 4);  // Enable MSAA
    
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Bouncing Balls", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync
    
    // Set up the coordinate system
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Enable blending and anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);  // Enable MSAA
}

// Initialize balls with random positions and velocities
void initBalls() {
    std::vector<Ball> balls(NUM_BALLS);
    
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> radiusDist(MIN_RADIUS, MAX_RADIUS);
    std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> velDist(-MAX_INITIAL_VELOCITY, MAX_INITIAL_VELOCITY);
    
    // Available radii
    const float radii[3] = {15.0f, 20.0f, 25.0f};
    
    for (int i = 0; i < NUM_BALLS; i++) {
        // Random radius from the three options
        int radiusIndex = gen() % 3;
        balls[i].radius = radii[radiusIndex];
        
        // Random position (keeping balls away from edges)
        balls[i].position.x = balls[i].radius + 
            posDist(gen) * (WINDOW_WIDTH - 2 * balls[i].radius);
        balls[i].position.y = balls[i].radius + 
            posDist(gen) * (WINDOW_HEIGHT - 2 * balls[i].radius);
        
        // Random velocity
        balls[i].velocity.x = velDist(gen);
        balls[i].velocity.y = velDist(gen);

        // Debug print
        std::cout << "Ball " << i << " initialized: pos=(" 
                  << balls[i].position.x << "," << balls[i].position.y 
                  << "), vel=(" << balls[i].velocity.x << "," << balls[i].velocity.y 
                  << "), radius=" << balls[i].radius << std::endl;
    }

    cl_int error = clEnqueueWriteBuffer(queue, ballBuffer, CL_TRUE, 0, 
                                       sizeof(Ball) * NUM_BALLS, balls.data(), 
                                       0, nullptr, nullptr);
    checkError(error, "writing initial ball data");
}

// Render function
void render() {
    // Clear the screen to black
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Read ball data back
    std::vector<Ball> balls(NUM_BALLS);
    cl_int error = clEnqueueReadBuffer(queue, ballBuffer, CL_TRUE, 0,
                                      sizeof(Ball) * NUM_BALLS, balls.data(),
                                      0, nullptr, nullptr);
    checkError(error, "reading ball data for rendering");
    
    // Define just three colors
    const float colors[3][3] = {
        {1.0f, 0.0f, 0.0f},  // Red
        {0.0f, 1.0f, 0.0f},  // Green
        {0.0f, 0.0f, 1.0f}   // Blue
    };
    
    // Enable anti-aliasing for smoother circles
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    // Draw each ball
    for (int i = 0; i < NUM_BALLS; i++) {
        const Ball& ball = balls[i];
        const int colorIndex = i % 3;  // Cycle through the 3 colors
        
        const int segments = 32;
        
        // Set ball color with alpha for smoother rendering
        glColor4f(colors[colorIndex][0], colors[colorIndex][1], colors[colorIndex][2], 0.9f);
        
        // Draw filled circle
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(ball.position.x, ball.position.y);  // Center
        for (int j = 0; j <= segments; j++) {
            float angle = 2.0f * M_PI * j / segments;
            float x = ball.position.x + cos(angle) * ball.radius;
            float y = ball.position.y + sin(angle) * ball.radius;
            glVertex2f(x, y);
        }
        glEnd();
        
        // Draw outline with the same color but full alpha
        glColor4f(colors[colorIndex][0], colors[colorIndex][1], colors[colorIndex][2], 1.0f);
        glLineWidth(2.0f);  // Make the outline slightly thicker
        glBegin(GL_LINE_LOOP);
        for (int j = 0; j < segments; j++) {
            float angle = 2.0f * M_PI * j / segments;
            float x = ball.position.x + cos(angle) * ball.radius;
            float y = ball.position.y + sin(angle) * ball.radius;
            glVertex2f(x, y);
        }
        glEnd();
    }
    
    glFinish();  // Ensure all rendering is complete before swap
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
    initOpenCL();
    initGraphics();  // This must come after initOpenCL
    initBalls();

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    auto lastFPSTime = lastTime;
    
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Cap delta time
        if (deltaTime > 0.05f) deltaTime = 0.05f;
        
        // Update FPS counter
        frameCount++;
        auto fpsDuration = std::chrono::duration<float>(currentTime - lastFPSTime).count();
        if (fpsDuration >= 1.0f) {
            float fps = frameCount / fpsDuration;
            std::cout << "FPS: " << fps << ", Delta Time: " << deltaTime << std::endl;
            frameCount = 0;
            lastFPSTime = currentTime;
        }

        // Clear collision count
        int zero = 0;
        cl_int error = clEnqueueWriteBuffer(queue, statsBuffer, CL_TRUE, 0, 
                                          sizeof(int), &zero, 0, nullptr, nullptr);
        checkError(error, "clearing stats buffer");

        // Update positions (GPU kernel)
        FLOAT2 boundaries = {static_cast<float>(WINDOW_WIDTH), 
                           static_cast<float>(WINDOW_HEIGHT)};
        error = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &ballBuffer);
        error |= clSetKernelArg(gpuKernel, 1, sizeof(float), &deltaTime);
        error |= clSetKernelArg(gpuKernel, 2, sizeof(FLOAT2), &boundaries);
        error |= clSetKernelArg(gpuKernel, 3, sizeof(int), &NUM_BALLS);
        checkError(error, "setting GPU kernel arguments");
        
        size_t globalSize = NUM_BALLS;
        error = clEnqueueNDRangeKernel(queue, gpuKernel, 1, nullptr, &globalSize, 
                                      nullptr, 0, nullptr, nullptr);
        checkError(error, "enqueueing GPU kernel");

        // Handle collisions (CPU kernel)
        error = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &ballBuffer);
        error |= clSetKernelArg(cpuKernel, 1, sizeof(int), &NUM_BALLS);
        error |= clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &statsBuffer);
        checkError(error, "setting CPU kernel arguments");
        
        error = clEnqueueNDRangeKernel(queue, cpuKernel, 1, nullptr, &globalSize, 
                                      nullptr, 0, nullptr, nullptr);
        checkError(error, "enqueueing CPU kernel");

        // Wait for kernels to complete
        clFinish(queue);

        // Read collision count
        int collisionCount;
        error = clEnqueueReadBuffer(queue, statsBuffer, CL_TRUE, 0,
                                  sizeof(int), &collisionCount, 0, nullptr, nullptr);
        if (collisionCount > 0) {
            std::cout << "Collisions this frame: " << collisionCount << std::endl;
        }

        // Render frame
        render();
        
        // Process events
        glfwPollEvents();
    }
    
    cleanup();
    return 0;
}
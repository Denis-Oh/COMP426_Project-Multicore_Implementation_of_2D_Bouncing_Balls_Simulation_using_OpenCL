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
const int NUM_BALLS = 50; 
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const float MIN_RADIUS = 10.0f;  
const float MAX_RADIUS = 20.0f;  
const float MAX_INITIAL_VELOCITY = 100.0f;  

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

    // Required for macOS to work correctly
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    
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
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1); // Note: Y-axis flipped for standard screen coordinates
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Enable blending for smoother rendering
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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
        // Ensure balls don't start too close to each other
        bool validPosition = false;
        while (!validPosition) {
            ball.position.x = posX(gen);
            ball.position.y = posY(gen);
            ball.radius = radius(gen);
            validPosition = true;
            
            // Check distance from other balls
            for (int i = 0; i < &ball - balls.data(); i++) {
                float dx = ball.position.x - balls[i].position.x;
                float dy = ball.position.y - balls[i].position.y;
                float minDist = ball.radius + balls[i].radius;
                if (dx * dx + dy * dy < minDist * minDist) {
                    validPosition = false;
                    break;
                }
            }
        }
        
        // Set random velocity
        ball.velocity.x = vel(gen);
        ball.velocity.y = vel(gen);
        
        // Ensure minimum speed
        float speed = sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
        if (speed < MAX_INITIAL_VELOCITY * 0.2f) {
            float scale = MAX_INITIAL_VELOCITY * 0.2f / speed;
            ball.velocity.x *= scale;
            ball.velocity.y *= scale;
        }
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
    
    // Draw each ball
    for (const auto& ball : balls) {
        const int segments = 32;
        glColor3f(1.0f, 0.5f, 0.2f);  // Orange color
        
        glBegin(GL_TRIANGLE_FAN);
        // Center vertex
        glVertex2f(ball.position.x, ball.position.y);
        
        // Circle vertices
        for (int i = 0; i <= segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = ball.position.x + cos(angle) * ball.radius;
            float y = ball.position.y + sin(angle) * ball.radius;
            glVertex2f(x, y);
        }
        glEnd();
        
        // Draw outline
        glColor3f(1.0f, 1.0f, 1.0f);  // White outline
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; i++) {
            float angle = 2.0f * M_PI * i / segments;
            float x = ball.position.x + cos(angle) * ball.radius;
            float y = ball.position.y + sin(angle) * ball.radius;
            glVertex2f(x, y);
        }
        glEnd();
    }
    
    // Debug: Draw window boundaries
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(0, 0);
    glVertex2f(WINDOW_WIDTH, 0);
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT);
    glVertex2f(0, WINDOW_HEIGHT);
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
#include <GLFW/glfw3.h>
#include <OpenCL/opencl.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <OpenGL/OpenGL.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>

// Structure definitions
struct Ball {
    cl_float4 position;  // x, y, radius, padding
    cl_float4 velocity;  // vx, vy, padding, padding
    cl_float4 color;     // r, g, b, padding
};

// Helper function to check OpenCL errors
void checkError(cl_int error, const char* location) {
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL error at " << location << ": " << error << std::endl;
        exit(error);
    }
}

// Helper function to load kernel source
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

// Buffer management class for double buffering
class BufferManager {
private:
    std::vector<cl_mem> positionBuffers;
    std::vector<cl_mem> velocityBuffers;
    std::vector<cl_mem> colorBuffers;
    int currentRead;
    int currentWrite;
    std::mutex mutex;
    
public:
    BufferManager(cl_context context, size_t numBalls) {
        positionBuffers.resize(2);
        velocityBuffers.resize(2);
        colorBuffers.resize(2);
        currentRead = 0;
        currentWrite = 1;
        
        cl_int error;
        for (int i = 0; i < 2; i++) {
            positionBuffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                sizeof(cl_float4) * numBalls, nullptr, &error);
            velocityBuffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(cl_float4) * numBalls, nullptr, &error);
            colorBuffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(cl_float4) * numBalls, nullptr, &error);
        }
    }
    
    void swap() {
        std::lock_guard<std::mutex> lock(mutex);
        std::swap(currentRead, currentWrite);
    }
    
    cl_mem getReadBuffer(int type) {
        std::lock_guard<std::mutex> lock(mutex);
        switch(type) {
            case 0: return positionBuffers[currentRead];
            case 1: return velocityBuffers[currentRead];
            case 2: return colorBuffers[currentRead];
            default: return nullptr;
        }
    }
    
    cl_mem getWriteBuffer(int type) {
        std::lock_guard<std::mutex> lock(mutex);
        switch(type) {
            case 0: return positionBuffers[currentWrite];
            case 1: return velocityBuffers[currentWrite];
            case 2: return colorBuffers[currentWrite];
            default: return nullptr;
        }
    }
};

class BallSimulation {
private:
    // OpenCL variables
    cl_context context;
    cl_command_queue cpuQueue;
    cl_command_queue gpuQueue;
    cl_program cpuProgram;
    cl_program gpuProgram;
    cl_kernel cpuPreparationKernel; 
    cl_kernel cpuStatsKernel;     
    cl_kernel gpuPhysicsKernel;
    cl_device_id cpuDevice;
    cl_device_id gpuDevice;
    size_t numBalls;
    const float gravity = -0.1f;
    
    // Buffer management
    BufferManager* bufferManager;
    cl_mem statsBuffer;
    cl_mem flagBuffer;
    
    // Simulation state
    std::vector<Ball> balls;
    bool computationComplete;
    bool displayComplete;
    std::mutex stateMutex;
    std::condition_variable computeCV; 
    std::condition_variable displayCV; 
    GLFWwindow* window;                
    
    // Window properties
    int windowWidth;
    int windowHeight;

    // Add function declaration
    void initBalls(int numBalls);         
    
public:
    BallSimulation(GLFWwindow* win, int width, int height) 
        : window(win), windowWidth(width), windowHeight(height), 
        computationComplete(false), displayComplete(true), numBalls(5) {
        initOpenCL();
        initBuffers();
        initBalls(numBalls);
    }
    
    void initOpenCL() {
        cl_int error;
        cl_uint numPlatforms;
        cl_platform_id platform;
        
        // Get platform
        error = clGetPlatformIDs(1, &platform, &numPlatforms);
        checkError(error, "clGetPlatformIDs");
        
        // Get CPU and GPU devices
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cpuDevice, nullptr);
        checkError(error, "clGetDeviceIDs CPU");
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpuDevice, nullptr);
        checkError(error, "clGetDeviceIDs GPU");
        
        // Create context
        CGLContextObj glContext = CGLGetCurrentContext();
        CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);
        
        cl_context_properties properties[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
            (cl_context_properties)shareGroup,
            0
        };
        
        context = clCreateContext(properties, 0, nullptr, nullptr, nullptr, &error);
        checkError(error, "clCreateContext");
        
        // Create command queues
        cpuQueue = clCreateCommandQueue(context, cpuDevice, 0, &error);
        checkError(error, "clCreateCommandQueue CPU");
        gpuQueue = clCreateCommandQueue(context, gpuDevice, 0, &error);
        checkError(error, "clCreateCommandQueue GPU");
        
        // Load and build programs
        std::string cpuSource = loadKernel("cpu_kernel.cl");
        const char* cpuSourcePtr = cpuSource.c_str();
        size_t cpuSourceSize = cpuSource.length();
        cpuProgram = clCreateProgramWithSource(context, 1, &cpuSourcePtr, &cpuSourceSize, &error);
        checkError(error, "clCreateProgramWithSource CPU");
        error = clBuildProgram(cpuProgram, 1, &cpuDevice, nullptr, nullptr, nullptr);
        checkError(error, "clBuildProgram CPU");
        
        std::string gpuSource = loadKernel("gpu_kernel.cl");
        const char* gpuSourcePtr = gpuSource.c_str();
        size_t gpuSourceSize = gpuSource.length();
        gpuProgram = clCreateProgramWithSource(context, 1, &gpuSourcePtr, &gpuSourceSize, &error);
        checkError(error, "clCreateProgramWithSource GPU");
        error = clBuildProgram(gpuProgram, 1, &gpuDevice, nullptr, nullptr, nullptr);
        checkError(error, "clBuildProgram GPU");
        
        // Create kernels
        cpuPreparationKernel = clCreateKernel(cpuProgram, "prepareBallData", &error);
        checkError(error, "clCreateKernel CPU Preparation");
        cpuStatsKernel = clCreateKernel(cpuProgram, "processStatistics", &error);
        checkError(error, "clCreateKernel CPU Stats");
        gpuPhysicsKernel = clCreateKernel(gpuProgram, "updateBallPhysics", &error);
        checkError(error, "clCreateKernel GPU Physics");
    }
    
    void initBuffers() {
        cl_int error;
        
        // Create buffer manager
        bufferManager = new BufferManager(context, numBalls);
        
        // Create statistics buffer
        statsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_int) * 4, nullptr, &error);
        checkError(error, "Create stats buffer");
        
        // Create synchronization flag buffer
        flagBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_int), nullptr, &error);
        checkError(error, "Create flag buffer");
    }

    void startSimulation() {
        // Launch display thread
        std::thread displayThread(&BallSimulation::displayLoop, this);
        
        // Launch computation thread
        std::thread computeThread(&BallSimulation::computeLoop, this);
        
        // Wait for threads to complete
        displayThread.join();
        computeThread.join();
    }
    
private:
    void displayLoop() {
        while (!shouldStop()) {
            // Wait for computation to complete
            waitForComputation();
            
            // Render current frame
            render();
            
            // Signal computation thread
            signalDisplayComplete();
        }
    }
    
    void computeLoop() {
        while (!shouldStop()) {
            // Wait for display to complete
            waitForDisplay();
            
            // Run CPU preparation kernel
            runCPUKernel();
            
            // Run GPU physics kernel
            runGPUKernel();
            
            // Run statistics kernel
            runStatsKernel();
            
            // Signal display thread
            signalComputationComplete();
        }
    }
    
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Read current positions and colors
        std::vector<cl_float4> positions(numBalls);
        std::vector<cl_float4> colors(numBalls);
        
        cl_int error = clEnqueueReadBuffer(cpuQueue, bufferManager->getReadBuffer(0), CL_TRUE, 0,
            sizeof(cl_float4) * numBalls, positions.data(), 0, nullptr, nullptr);
        checkError(error, "Read positions for render");
        
        error = clEnqueueReadBuffer(cpuQueue, bufferManager->getReadBuffer(2), CL_TRUE, 0,
            sizeof(cl_float4) * numBalls, colors.data(), 0, nullptr, nullptr);
        checkError(error, "Read colors for render");
        
        // Draw balls
        for (size_t i = 0; i < numBalls; ++i) {
            glColor3f(colors[i].s[0], colors[i].s[1], colors[i].s[2]);
            
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(positions[i].s[0], positions[i].s[1]);
            for (int angle = 0; angle <= 360; angle += 10) {
                float radian = angle * 3.14159f / 180.0f;
                float x = positions[i].s[0] + cos(radian) * positions[i].s[2];
                float y = positions[i].s[1] + sin(radian) * positions[i].s[2];
                glVertex2f(x, y);
            }
            glEnd();
        }
        
        glfwSwapBuffers(window);
    }

    void runCPUKernel() {
        cl_int error;
        
        // Get buffer references
        cl_mem readBuf = bufferManager->getReadBuffer(0);
        cl_mem writeBuf = bufferManager->getWriteBuffer(0);
        
        // Set kernel arguments for preparation kernel
        error = clSetKernelArg(cpuPreparationKernel, 0, sizeof(cl_mem), &readBuf);
        error |= clSetKernelArg(cpuPreparationKernel, 1, sizeof(cl_mem), &writeBuf);
        error |= clSetKernelArg(cpuPreparationKernel, 2, sizeof(cl_mem), &flagBuffer);
        error |= clSetKernelArg(cpuPreparationKernel, 3, sizeof(cl_int), &numBalls);
        error |= clSetKernelArg(cpuPreparationKernel, 4, sizeof(cl_float), &windowWidth);
        error |= clSetKernelArg(cpuPreparationKernel, 5, sizeof(cl_float), &windowHeight);
        checkError(error, "Set CPU preparation kernel args");
        
        // Execute kernel
        size_t globalSize = numBalls;
        error = clEnqueueNDRangeKernel(cpuQueue, cpuPreparationKernel, 1, nullptr,
            &globalSize, nullptr, 0, nullptr, nullptr);
        checkError(error, "Enqueue CPU preparation kernel");
    }
    
    void runGPUKernel() {
        cl_int error;
        
        // Get buffer references
        cl_mem posBuf = bufferManager->getWriteBuffer(0);
        cl_mem velBuf = bufferManager->getWriteBuffer(1);
        cl_mem colBuf = bufferManager->getWriteBuffer(2);
        
        // Set kernel arguments
        error = clSetKernelArg(gpuPhysicsKernel, 0, sizeof(cl_mem), &posBuf);
        error |= clSetKernelArg(gpuPhysicsKernel, 1, sizeof(cl_mem), &velBuf);
        error |= clSetKernelArg(gpuPhysicsKernel, 2, sizeof(cl_mem), &colBuf);
        error |= clSetKernelArg(gpuPhysicsKernel, 3, sizeof(cl_int), &numBalls);
        error |= clSetKernelArg(gpuPhysicsKernel, 4, sizeof(cl_float), &gravity);
        error |= clSetKernelArg(gpuPhysicsKernel, 5, sizeof(cl_float), &windowWidth);
        error |= clSetKernelArg(gpuPhysicsKernel, 6, sizeof(cl_float), &windowHeight);
        error |= clSetKernelArg(gpuPhysicsKernel, 7, sizeof(cl_mem), &flagBuffer);
        checkError(error, "Set GPU physics kernel args");
        
        // Execute kernel
        size_t globalSize = numBalls * 32; // THREADS_PER_BALL from gpu_kernel.cl
        size_t localSize = 256; // LOCAL_SIZE from gpu_kernel.cl
        error = clEnqueueNDRangeKernel(gpuQueue, gpuPhysicsKernel, 1, nullptr,
            &globalSize, &localSize, 0, nullptr, nullptr);
        checkError(error, "Enqueue GPU physics kernel");
        
        // Swap buffers after computation
        bufferManager->swap();
    }
    
    void runStatsKernel() {
        cl_int error;
        
        // Reset statistics buffer
        cl_int zeroStats[4] = {0, 0, 0, 0};
        error = clEnqueueWriteBuffer(cpuQueue, statsBuffer, CL_TRUE, 0,
            sizeof(cl_int) * 4, zeroStats, 0, nullptr, nullptr);
        checkError(error, "Reset stats buffer");
        
        // Get buffer reference
        cl_mem readBuf = bufferManager->getReadBuffer(0);
        
        // Set kernel arguments
        error = clSetKernelArg(cpuStatsKernel, 0, sizeof(cl_mem), &readBuf);
        error |= clSetKernelArg(cpuStatsKernel, 1, sizeof(cl_mem), &statsBuffer);
        error |= clSetKernelArg(cpuStatsKernel, 2, sizeof(cl_int), &numBalls);
        checkError(error, "Set CPU stats kernel args");
        
        // Execute kernel
        size_t globalSize = numBalls;
        error = clEnqueueNDRangeKernel(cpuQueue, cpuStatsKernel, 1, nullptr,
            &globalSize, nullptr, 0, nullptr, nullptr);
        checkError(error, "Enqueue CPU stats kernel");
    }

    // Synchronization methods
    void waitForComputation() {
        std::unique_lock<std::mutex> lock(stateMutex);
        computeCV.wait(lock, [this] { return computationComplete; });
        computationComplete = false;
    }

    void waitForDisplay() {
        std::unique_lock<std::mutex> lock(stateMutex);
        displayCV.wait(lock, [this] { return displayComplete; });
        displayComplete = false;
    }

    void signalComputationComplete() {
        std::lock_guard<std::mutex> lock(stateMutex);
        computationComplete = true;
        computeCV.notify_one();
    }

    void signalDisplayComplete() {
        std::lock_guard<std::mutex> lock(stateMutex);
        displayComplete = true;
        displayCV.notify_one();
    }

    bool shouldStop() {
        return glfwWindowShouldClose(window);
    }
};

void BallSimulation::initBalls(int numBalls) {
    balls.resize(numBalls);
    std::vector<cl_float4> positions(numBalls);
    std::vector<cl_float4> velocities(numBalls);
    std::vector<cl_float4> colors(numBalls);
    
    srand(static_cast<unsigned int>(time(0)));
    float radius_options[] = {50.0f, 100.0f, 150.0f};
    
    for (int i = 0; i < numBalls; ++i) {
        float radius = radius_options[rand() % 3];
        
        // Initialize positions
        positions[i].s[0] = radius + (rand() % (int)(windowWidth - 2 * radius));  // x
        positions[i].s[1] = radius + (rand() % (int)(windowHeight - 2 * radius)); // y
        positions[i].s[2] = radius;                                               // radius
        positions[i].s[3] = 0.0f;                                                // padding
        
        // Initialize velocities
        velocities[i].s[0] = (float)(rand() % 200 - 100) / 50.0f;  // vx
        velocities[i].s[1] = (float)(rand() % 200 - 100) / 50.0f;  // vy
        velocities[i].s[2] = 0.0f;                                 // padding
        velocities[i].s[3] = 0.0f;                                 // padding
        
        // Initialize colors
        colors[i].s[0] = (i % 3 == 0) ? 1.0f : 0.0f;  // r
        colors[i].s[1] = (i % 3 == 1) ? 1.0f : 0.0f;  // g
        colors[i].s[2] = (i % 3 == 2) ? 1.0f : 0.0f;  // b
        colors[i].s[3] = 0.0f;                        // alpha
        
        balls[i].position = positions[i];
        balls[i].velocity = velocities[i];
        balls[i].color = colors[i];
    }
    
    // Initialize buffers with ball data
    cl_int error;
    error = clEnqueueWriteBuffer(cpuQueue, bufferManager->getWriteBuffer(0), CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, positions.data(), 0, nullptr, nullptr);
    checkError(error, "Write positions");
    
    error = clEnqueueWriteBuffer(cpuQueue, bufferManager->getWriteBuffer(1), CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, velocities.data(), 0, nullptr, nullptr);
    checkError(error, "Write velocities");
    
    error = clEnqueueWriteBuffer(cpuQueue, bufferManager->getWriteBuffer(2), CL_TRUE, 0,
        sizeof(cl_float4) * numBalls, colors.data(), 0, nullptr, nullptr);
    checkError(error, "Write colors");
}

int main() {
    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, "OpenCL Bouncing Balls", NULL, NULL);
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
    
    // Initialize simulation with window
    BallSimulation simulation(window, 1200, 900);
    
    // Start simulation
    simulation.startSimulation();
    
    // Cleanup
    glfwTerminate();
    return 0;
}
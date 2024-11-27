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
        std::string errorString;
        switch(error) {
            case CL_DEVICE_NOT_FOUND: errorString = "CL_DEVICE_NOT_FOUND"; break;
            case CL_DEVICE_NOT_AVAILABLE: errorString = "CL_DEVICE_NOT_AVAILABLE"; break;
            case CL_COMPILER_NOT_AVAILABLE: errorString = "CL_COMPILER_NOT_AVAILABLE"; break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: errorString = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
            case CL_OUT_OF_RESOURCES: errorString = "CL_OUT_OF_RESOURCES"; break;
            case CL_OUT_OF_HOST_MEMORY: errorString = "CL_OUT_OF_HOST_MEMORY"; break;
            case CL_INVALID_VALUE: errorString = "CL_INVALID_VALUE"; break;
            case CL_INVALID_DEVICE_TYPE: errorString = "CL_INVALID_DEVICE_TYPE"; break;
            case CL_INVALID_PLATFORM: errorString = "CL_INVALID_PLATFORM"; break;
            case CL_INVALID_DEVICE: errorString = "CL_INVALID_DEVICE"; break;
            case CL_INVALID_CONTEXT: errorString = "CL_INVALID_CONTEXT"; break;
            case CL_INVALID_QUEUE_PROPERTIES: errorString = "CL_INVALID_QUEUE_PROPERTIES"; break;
            case CL_INVALID_COMMAND_QUEUE: errorString = "CL_INVALID_COMMAND_QUEUE"; break;
            case CL_INVALID_HOST_PTR: errorString = "CL_INVALID_HOST_PTR"; break;
            case CL_INVALID_MEM_OBJECT: errorString = "CL_INVALID_MEM_OBJECT"; break;
            case CL_INVALID_PROGRAM: errorString = "CL_INVALID_PROGRAM"; break;
            case CL_INVALID_PROGRAM_EXECUTABLE: errorString = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
            case CL_INVALID_KERNEL_NAME: errorString = "CL_INVALID_KERNEL_NAME"; break;
            case CL_INVALID_KERNEL_DEFINITION: errorString = "CL_INVALID_KERNEL_DEFINITION"; break;
            case CL_INVALID_KERNEL: errorString = "CL_INVALID_KERNEL"; break;
            case CL_INVALID_ARG_INDEX: errorString = "CL_INVALID_ARG_INDEX"; break;
            case CL_INVALID_ARG_VALUE: errorString = "CL_INVALID_ARG_VALUE"; break;
            case CL_INVALID_ARG_SIZE: errorString = "CL_INVALID_ARG_SIZE"; break;
            case CL_INVALID_KERNEL_ARGS: errorString = "CL_INVALID_KERNEL_ARGS"; break;
            case CL_INVALID_WORK_DIMENSION: errorString = "CL_INVALID_WORK_DIMENSION"; break;
            case CL_INVALID_WORK_GROUP_SIZE: errorString = "CL_INVALID_WORK_GROUP_SIZE"; break;
            case CL_INVALID_WORK_ITEM_SIZE: errorString = "CL_INVALID_WORK_ITEM_SIZE"; break;
            default: errorString = "UNKNOWN ERROR";
        }
        std::cerr << "OpenCL error at " << location << ": " << errorString << " (" << error << ")" << std::endl;
        throw std::runtime_error("OpenCL error: " + errorString + " at " + location);
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
    std::vector<cl_mem> stateBuffers;  // Store complete state buffers
    int currentRead;
    int currentWrite;
    std::mutex mutex;
    cl_command_queue queue;
    
public:
    BufferManager(cl_context context, cl_command_queue q, size_t numBalls) 
        : queue(q) {
        stateBuffers.resize(2);
        currentRead = 0;
        currentWrite = 1;
        
        cl_int error;
        for (int i = 0; i < 2; i++) {
            stateBuffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                sizeof(Ball) * numBalls, nullptr, &error);
            checkError(error, "Create state buffer");
        }
    }
    
    ~BufferManager() {
        for (auto buffer : stateBuffers) {
            clReleaseMemObject(buffer);
        }
    }
    
    void swap() {
        std::lock_guard<std::mutex> lock(mutex);
        // Ensure all operations are complete before swapping
        cl_int error = clFinish(queue);
        checkError(error, "Finish queue before buffer swap");
        std::swap(currentRead, currentWrite);
    }
    
    cl_mem getReadBuffer() {
        std::lock_guard<std::mutex> lock(mutex);
        return stateBuffers[currentRead];
    }
    
    cl_mem getWriteBuffer() {
        std::lock_guard<std::mutex> lock(mutex);
        return stateBuffers[currentWrite];
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
    GLFWwindow* renderWindow;
    std::thread::id mainThreadId;                
    
    // Window properties
    int windowWidth;
    int windowHeight;

    // Add function declaration
    void initBalls(int numBalls);         
    
public:
    BallSimulation(GLFWwindow* win, int width, int height) 
        : window(win), windowWidth(width), windowHeight(height),
          computationComplete(false), displayComplete(true), numBalls(5) {
        mainThreadId = std::this_thread::get_id();
        renderWindow = window;
        
        std::cout << "Initializing OpenCL..." << std::endl;
        try {
            initOpenCL();
            std::cout << "OpenCL initialized successfully" << std::endl;
            
            std::cout << "Initializing buffers..." << std::endl;
            initBuffers();
            std::cout << "Buffers initialized successfully" << std::endl;
            
            std::cout << "Initializing balls..." << std::endl;
            initBalls(numBalls);
            std::cout << "Balls initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in BallSimulation constructor: " << e.what() << std::endl;
            throw;
        }
    }
    
    void initOpenCL() {
        cl_int error;
        cl_uint numPlatforms;
        cl_platform_id platform;
        
        // Get Apple platform
        error = clGetPlatformIDs(1, &platform, &numPlatforms);
        checkError(error, "clGetPlatformIDs");
        
        // Get the Apple device (unified CPU/GPU on M1)
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &cpuDevice, nullptr);
        checkError(error, "clGetDeviceIDs");
        
        // Use the same device for GPU operations
        gpuDevice = cpuDevice;
        
        // Print device info (helpful for your report)
        char deviceName[128];
        clGetDeviceInfo(cpuDevice, CL_DEVICE_NAME, 128, deviceName, nullptr);
        std::cout << "Using device: " << deviceName << " for both CPU and GPU operations\n";
        
        // Create context (rest remains the same)
        CGLContextObj glContext = CGLGetCurrentContext();
        CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);
        
        cl_context_properties properties[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
            (cl_context_properties)shareGroup,
            0
        };
        
        context = clCreateContext(properties, 0, nullptr, nullptr, nullptr, &error);
        checkError(error, "clCreateContext");
        
        // Create separate command queues (for logical separation)
        cpuQueue = clCreateCommandQueue(context, cpuDevice, 0, &error);
        checkError(error, "clCreateCommandQueue CPU");
        gpuQueue = clCreateCommandQueue(context, cpuDevice, 0, &error);
        checkError(error, "clCreateCommandQueue GPU");
        
        // Load and build both programs as before
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
        
        cl_int buildError;
        size_t logSize;
        char* buildLog;

        // Check CPU program build
        buildError = clGetProgramBuildInfo(cpuProgram, cpuDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        if (logSize > 1) {
            buildLog = new char[logSize];
            clGetProgramBuildInfo(cpuProgram, cpuDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
            std::cout << "CPU Program build log:" << std::endl << buildLog << std::endl;
            delete[] buildLog;
        }

        // Check GPU program build
        buildError = clGetProgramBuildInfo(gpuProgram, gpuDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        if (logSize > 1) {
            buildLog = new char[logSize];
            clGetProgramBuildInfo(gpuProgram, gpuDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
            std::cout << "GPU Program build log:" << std::endl << buildLog << std::endl;
            delete[] buildLog;
        }

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
        bufferManager = new BufferManager(context, cpuQueue, numBalls);
        
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
        if (std::this_thread::get_id() != mainThreadId) {
            throw std::runtime_error("Simulation must be started from the main thread");
        }

        // Launch computation thread
        std::thread computeThread(&BallSimulation::computeLoop, this);
        
        // Run display loop in the main thread
        displayLoop();
        
        // Wait for compute thread to complete
        computeThread.join();
    }


    ~BallSimulation() {
        std::cout << "Cleaning up BallSimulation..." << std::endl;
        
        // Release OpenCL objects
        if (cpuQueue) clReleaseCommandQueue(cpuQueue);
        if (gpuQueue) clReleaseCommandQueue(gpuQueue);
        if (cpuPreparationKernel) clReleaseKernel(cpuPreparationKernel);
        if (cpuStatsKernel) clReleaseKernel(cpuStatsKernel);
        if (gpuPhysicsKernel) clReleaseKernel(gpuPhysicsKernel);
        if (cpuProgram) clReleaseProgram(cpuProgram);
        if (gpuProgram) clReleaseProgram(gpuProgram);
        if (context) clReleaseContext(context);
        if (statsBuffer) clReleaseMemObject(statsBuffer);
        if (flagBuffer) clReleaseMemObject(flagBuffer);
        
        delete bufferManager;
        
        std::cout << "BallSimulation cleanup complete" << std::endl;
    }
    
private:
    void displayLoop() {
        try {
            while (!shouldStop()) {
                waitForComputation();
                
                // Make sure we're using the correct context
                glfwMakeContextCurrent(renderWindow);
                
                render();
                
                // Process events
                glfwPollEvents();
                
                signalDisplayComplete();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in display loop: " << e.what() << std::endl;
            glfwSetWindowShouldClose(renderWindow, GLFW_TRUE);
        }
    }

    void computeLoop() {
        try {
            while (!shouldStop()) {
                waitForDisplay();
                
                // Run kernels in sequence
                runCPUKernel();
                runGPUKernel();
                runStatsKernel();
                
                // Swap buffers after computation is complete
                bufferManager->swap();
                
                signalComputationComplete();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in compute loop: " << e.what() << std::endl;
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    
    void render() {
        // Finish any pending OpenCL operations before rendering
        cl_int error = clFinish(cpuQueue);
        checkError(error, "Finish CPU queue before render");
        error = clFinish(gpuQueue);
        checkError(error, "Finish GPU queue before render");

        glClear(GL_COLOR_BUFFER_BIT);
        
        // Read current ball states
        std::vector<Ball> balls(numBalls);
        error = clEnqueueReadBuffer(cpuQueue, bufferManager->getReadBuffer(), 
            CL_TRUE, 0, sizeof(Ball) * numBalls, balls.data(), 0, nullptr, nullptr);
        checkError(error, "Read balls for render");
        
        // Set up the view
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Draw balls
        for (size_t i = 0; i < numBalls; ++i) {
            const Ball& ball = balls[i];
            glColor3f(ball.color.s[0], ball.color.s[1], ball.color.s[2]);
            
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(ball.position.s[0], ball.position.s[1]);
            for (int angle = 0; angle <= 360; angle += 10) {
                float radian = angle * 3.14159f / 180.0f;
                float x = ball.position.s[0] + cos(radian) * ball.position.s[2];
                float y = ball.position.s[1] + sin(radian) * ball.position.s[2];
                glVertex2f(x, y);
            }
            glEnd();
        }
        // Ensure rendering is complete before proceeding
        glFinish();
        glfwSwapBuffers(renderWindow);
    }

    void runCPUKernel() {
        cl_int error;
        
        cl_mem readBuf = bufferManager->getReadBuffer();
        cl_mem writeBuf = bufferManager->getWriteBuffer();
        
        error = clSetKernelArg(cpuPreparationKernel, 0, sizeof(cl_mem), &readBuf);
        error |= clSetKernelArg(cpuPreparationKernel, 1, sizeof(cl_mem), &writeBuf);
        error |= clSetKernelArg(cpuPreparationKernel, 2, sizeof(cl_mem), &flagBuffer);
        error |= clSetKernelArg(cpuPreparationKernel, 3, sizeof(cl_int), &numBalls);
        error |= clSetKernelArg(cpuPreparationKernel, 4, sizeof(cl_float), &windowWidth);
        error |= clSetKernelArg(cpuPreparationKernel, 5, sizeof(cl_float), &windowHeight);
        checkError(error, "Set CPU preparation kernel args");
        
        size_t globalSize = numBalls;
        error = clEnqueueNDRangeKernel(cpuQueue, cpuPreparationKernel, 1, nullptr,
            &globalSize, nullptr, 0, nullptr, nullptr);
        checkError(error, "Enqueue CPU preparation kernel");
    }
    
    void runGPUKernel() {
        cl_int error;
        
        // Clear completion flag
        cl_int zero = 0;
        error = clEnqueueWriteBuffer(gpuQueue, flagBuffer, CL_TRUE, 0, 
            sizeof(cl_int), &zero, 0, nullptr, nullptr);
        checkError(error, "Clear completion flag");

        // barrier to ensure flag is cleared before kernel execution
        error = clFinish(gpuQueue);
        checkError(error, "Finish queue after flag clear");
        
        cl_mem stateBuf = bufferManager->getWriteBuffer();
        
        error = clSetKernelArg(gpuPhysicsKernel, 0, sizeof(cl_mem), &stateBuf);
        error |= clSetKernelArg(gpuPhysicsKernel, 1, sizeof(cl_int), &numBalls);
        error |= clSetKernelArg(gpuPhysicsKernel, 2, sizeof(cl_float), &gravity);
        error |= clSetKernelArg(gpuPhysicsKernel, 3, sizeof(cl_float), &windowWidth);
        error |= clSetKernelArg(gpuPhysicsKernel, 4, sizeof(cl_float), &windowHeight);
        error |= clSetKernelArg(gpuPhysicsKernel, 5, sizeof(cl_mem), &flagBuffer);
        checkError(error, "Set GPU physics kernel args");
        
        // Get device capabilities
        size_t maxWorkGroupSize;
        error = clGetKernelWorkGroupInfo(gpuPhysicsKernel, gpuDevice, 
            CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);
        checkError(error, "Get kernel work group info");
        
        size_t localSize = std::min(maxWorkGroupSize, (size_t)64);
        size_t globalSize = ((numBalls + localSize - 1) / localSize) * localSize;
        
        error = clEnqueueNDRangeKernel(gpuQueue, gpuPhysicsKernel, 1, nullptr,
            &globalSize, &localSize, 0, nullptr, nullptr);
        checkError(error, "Enqueue GPU physics kernel");

        // barrier after kernel execution
        error = clFinish(gpuQueue);
        checkError(error, "Finish queue after kernel");
    }

    void runStatsKernel() {
        cl_int error;
        
        // Reset statistics buffer
        cl_int zeroStats[4] = {0, 0, 0, 0};
        error = clEnqueueWriteBuffer(cpuQueue, statsBuffer, CL_TRUE, 0,
            sizeof(cl_int) * 4, zeroStats, 0, nullptr, nullptr);
        checkError(error, "Reset stats buffer");
        
        cl_mem readBuf = bufferManager->getReadBuffer();
        
        error = clSetKernelArg(cpuStatsKernel, 0, sizeof(cl_mem), &readBuf);
        error |= clSetKernelArg(cpuStatsKernel, 1, sizeof(cl_mem), &statsBuffer);
        error |= clSetKernelArg(cpuStatsKernel, 2, sizeof(cl_int), &numBalls);
        checkError(error, "Set CPU stats kernel args");
        
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
    std::vector<Ball> initialBalls(numBalls);
    
    srand(static_cast<unsigned int>(time(0)));
    float radius_options[] = {50.0f, 100.0f, 150.0f};
    
    for (int i = 0; i < numBalls; ++i) {
        Ball& ball = initialBalls[i];
        float radius = radius_options[rand() % 3];
        
        // Position
        ball.position.s[0] = radius + (rand() % (int)(windowWidth - 2 * radius));
        ball.position.s[1] = radius + (rand() % (int)(windowHeight - 2 * radius));
        ball.position.s[2] = radius;
        ball.position.s[3] = 0.0f;
        
        // Velocity
        ball.velocity.s[0] = (float)(rand() % 200 - 100) / 50.0f;
        ball.velocity.s[1] = (float)(rand() % 200 - 100) / 50.0f;
        ball.velocity.s[2] = 0.0f;
        ball.velocity.s[3] = 0.0f;
        
        // Color
        ball.color.s[0] = (i % 3 == 0) ? 1.0f : 0.0f;
        ball.color.s[1] = (i % 3 == 1) ? 1.0f : 0.0f;
        ball.color.s[2] = (i % 3 == 2) ? 1.0f : 0.0f;
        ball.color.s[3] = 1.0f;
    }
    
    // Write initial data to both buffers
    cl_int error = clEnqueueWriteBuffer(cpuQueue, bufferManager->getWriteBuffer(), 
        CL_TRUE, 0, sizeof(Ball) * numBalls, initialBalls.data(), 0, nullptr, nullptr);
    checkError(error, "Write initial ball data");
    
    error = clEnqueueWriteBuffer(cpuQueue, bufferManager->getReadBuffer(), 
        CL_TRUE, 0, sizeof(Ball) * numBalls, initialBalls.data(), 0, nullptr, nullptr);
    checkError(error, "Write initial ball data to read buffer");
}

int main() {
    std::cout << "Starting program..." << std::endl;
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    std::cout << "GLFW initialized successfully" << std::endl;
    
    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, "OpenCL Bouncing Balls", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    std::cout << "GLFW window created successfully" << std::endl;
    
    glfwMakeContextCurrent(window);
    
    // Initialize OpenGL
    std::cout << "Initializing OpenGL..." << std::endl;
    glViewport(0, 0, 1200, 900);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1200, 900, 0, -1, 1);  // Note: Modified to match window coordinates
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    std::cout << "OpenGL initialized successfully" << std::endl;
    
    try {
        std::cout << "Creating simulation..." << std::endl;
        BallSimulation simulation(window, 1200, 900);
        std::cout << "Simulation created successfully" << std::endl;
        
        std::cout << "Starting simulation..." << std::endl;
        simulation.startSimulation();
        std::cout << "Simulation completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during simulation: " << e.what() << std::endl;
    }
    
    std::cout << "Cleaning up..." << std::endl;
    glfwTerminate();
    std::cout << "Program ended successfully" << std::endl;
    return 0;
}
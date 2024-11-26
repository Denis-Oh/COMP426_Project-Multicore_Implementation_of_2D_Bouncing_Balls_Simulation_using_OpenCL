// Structures for ball state management
typedef struct {
    float4 position;  // x, y, radius, padding
    float4 velocity;  // vx, vy, padding, padding
    float4 color;     // r, g, b, alpha
} BallState;

// Kernel for data preparation and validation
__kernel void prepareBallData(
    __global BallState* inputStates,
    __global BallState* outputStates,
    __global int* stateFlags,
    const int numBalls,
    const float windowWidth,
    const float windowHeight
) {
    int idx = get_global_id(0);
    if (idx >= numBalls) return;

    // Copy state
    BallState state = inputStates[idx];
    
    // Validate position bounds
    float radius = state.position.z;
    float minX = radius;
    float maxX = windowWidth - radius;
    float minY = radius;
    float maxY = windowHeight - radius;
    
    // Clamp positions to valid ranges
    state.position.x = clamp(state.position.x, minX, maxX);
    state.position.y = clamp(state.position.y, minY, maxY);
    
    // Validate velocities (prevent extreme values)
    float maxVelocity = 10.0f;
    state.velocity.x = clamp(state.velocity.x, -maxVelocity, maxVelocity);
    state.velocity.y = clamp(state.velocity.y, -maxVelocity, maxVelocity);
    
    // Store validated state
    outputStates[idx] = state;
    
    // Set validation flag for this ball
    stateFlags[idx] = 1;
}

// Kernel for post-processing and statistics
__kernel void processStatistics(
    __global BallState* states,
    __global int* statsArray, 
    const int numBalls
) {
    int idx = get_global_id(0);
    if (idx >= numBalls) return;
    
    BallState state = states[idx];
    float mass = state.position.z * 0.1f;  // Simple mass calculation based on radius
    
    // Calculate velocities and energies
    float velocity = sqrt(state.velocity.x * state.velocity.x + 
                        state.velocity.y * state.velocity.y);
    float kineticEnergy = 0.5f * mass * velocity * velocity;
    float potentialEnergy = mass * 9.81f * state.position.y;
    float totalEnergy = kineticEnergy + potentialEnergy;
    
    // Convert float values to integers (multiply by 1000 to preserve 3 decimal places)
    int velocityInt = (int)(velocity * 1000);
    int energyInt = (int)(totalEnergy * 1000);
    
    // Atomic operations using integers
    atomic_add(&statsArray[0], velocityInt);     // Sum of velocities
    atomic_max(&statsArray[1], velocityInt);     // Max velocity
    atomic_add(&statsArray[2], energyInt);       // Total energy
    atomic_add(&statsArray[3], 1);              // Counter for averaging
}
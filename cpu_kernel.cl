typedef struct {
    float4 position;  // x, y, radius, padding
    float4 velocity;  // vx, vy, padding, padding
    float4 color;     // r, g, b, alpha
} BallState;

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
    
    // Copy input state to output state
    BallState state = inputStates[idx];
    
    // Ensure position is within bounds and fix if necessary
    float radius = state.position.s2;
    
    // Clamp positions to stay within window bounds
    state.position.s0 = clamp(state.position.s0, radius, windowWidth - radius);
    state.position.s1 = clamp(state.position.s1, radius, windowHeight - radius);
    
    // Write validated state to output buffer
    outputStates[idx] = state;
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
    
    BallState localState = states[idx];
    float mass = localState.position.s2 * 0.1f;  // Simple mass calculation based on radius
    
    // Calculate velocities and energies
    float velocity = sqrt(localState.velocity.s0 * localState.velocity.s0 + 
                        localState.velocity.s1 * localState.velocity.s1);
    float kineticEnergy = 0.5f * mass * velocity * velocity;
    float potentialEnergy = mass * 9.81f * localState.position.s1;
    float totalEnergy = kineticEnergy + potentialEnergy;
    
    // Convert float values to integers (multiply by 1000 to preserve 3 decimal places)
    int velocityInt = (int)(velocity * 1000);
    int energyInt = (int)(totalEnergy * 1000);
    
    // Atomic operations using integers
    atomic_add(&statsArray[0], velocityInt);     // Sum of velocities
    atomic_max(&statsArray[1], velocityInt);     // Max velocity
    atomic_add(&statsArray[2], energyInt);       // Total energy
    atomic_add(&statsArray[3], 1);               // Counter for averaging
}
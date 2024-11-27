typedef struct {
    float4 position;  // x, y, radius, padding
    float4 velocity;  // vx, vy, padding, padding
    float4 color;     // r, g, b, alpha
} BallState;

__kernel void updateBallPhysics(
    __global BallState* states,
    const int numBalls,
    const float gravity,
    const float windowWidth,
    const float windowHeight,
    __global int* computationComplete
) {
    int idx = get_global_id(0);
    if (idx >= numBalls) return;
    
    // Load the ball state
    BallState state = states[idx];
    float radius = state.position.s2;
    
    // Time step for physics (adjust as needed)
    const float dt = 0.016f;  // ~60 FPS
    
    // Update velocity with gravity
    state.velocity.s1 += gravity * dt;
    
    // Update position using current velocity
    state.position.s0 += state.velocity.s0 * dt;
    state.position.s1 += state.velocity.s1 * dt;
    
    // Handle collisions with walls and apply damping
    const float restitution = 0.8f;  // Energy retention on collision
    if (state.position.s0 - radius < 0) {
        state.position.s0 = radius;
        state.velocity.s0 = fabs(state.velocity.s0) * restitution;
    } else if (state.position.s0 + radius > windowWidth) {
        state.position.s0 = windowWidth - radius;
        state.velocity.s0 = -fabs(state.velocity.s0) * restitution;
    }
    
    if (state.position.s1 - radius < 0) {
        state.position.s1 = radius;
        state.velocity.s1 = fabs(state.velocity.s1) * restitution;
    } else if (state.position.s1 + radius > windowHeight) {
        state.position.s1 = windowHeight - radius;
        state.velocity.s1 = -fabs(state.velocity.s1) * restitution;
    }
    
    // Write back updated state
    states[idx] = state;
    
    // Set completion flag
    if (idx == 0) {
        *computationComplete = 1;
    }
}
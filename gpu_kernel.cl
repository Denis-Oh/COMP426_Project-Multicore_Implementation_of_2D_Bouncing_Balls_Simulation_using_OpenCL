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
    int global_id = get_global_id(0);
    if (global_id >= numBalls) return;
    
    // Load the ball state
    BallState state = states[global_id];
    
    // Update velocity and position
    state.velocity.y += gravity;
    state.position.x += state.velocity.x;
    state.position.y += state.velocity.y;
    
    // Boundary collisions
    float dampening = 0.95f;
    float radius = state.position.z;
    
    // X boundaries
    if (state.position.x - radius < 0) {
        state.position.x = radius;
        state.velocity.x = -state.velocity.x * dampening;
    } else if (state.position.x + radius > windowWidth) {
        state.position.x = windowWidth - radius;
        state.velocity.x = -state.velocity.x * dampening;
    }
    
    // Y boundaries
    if (state.position.y - radius < 0) {
        state.position.y = radius;
        state.velocity.y = -state.velocity.y * dampening;
    } else if (state.position.y + radius > windowHeight) {
        state.position.y = windowHeight - radius;
        state.velocity.y = -state.velocity.y * dampening;
    }
    
    // Ball-to-ball collisions
    for (int i = 0; i < numBalls; i++) {
        if (i == global_id) continue;
        
        BallState other = states[i];
        float dx = other.position.x - state.position.x;
        float dy = other.position.y - state.position.y;
        float distance = sqrt(dx * dx + dy * dy);
        float minDist = state.position.z + other.position.z;
        
        if (distance < minDist && distance > 0.0f) {
            float nx = dx / distance;
            float ny = dy / distance;
            
            if (global_id < i) {
                float overlap = minDist - distance;
                state.position.x -= nx * overlap * 0.5f;
                state.position.y -= ny * overlap * 0.5f;
            }
            
            float rvx = other.velocity.x - state.velocity.x;
            float rvy = other.velocity.y - state.velocity.y;
            float velAlongNormal = rvx * nx + rvy * ny;
            
            if (velAlongNormal < 0) {
                float restitution = 0.8f;
                float j = -(1.0f + restitution) * velAlongNormal;
                state.velocity.x += j * nx;
                state.velocity.y += j * ny;
            }
        }
    }
    
    // Speed limiting
    float maxSpeed = 5.0f;
    float speedSq = state.velocity.x * state.velocity.x + state.velocity.y * state.velocity.y;
    if (speedSq > maxSpeed * maxSpeed) {
        float scale = maxSpeed / sqrt(speedSq);
        state.velocity.x *= scale;
        state.velocity.y *= scale;
    }
    
    // Write back results
    states[global_id] = state;
    
    // Last thread sets completion flag
    if (global_id == numBalls - 1) {
        *computationComplete = 1;
    }
}
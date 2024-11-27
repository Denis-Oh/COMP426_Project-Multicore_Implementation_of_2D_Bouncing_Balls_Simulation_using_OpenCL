// Reuse Ball structure definition for consistency
typedef struct {
    float2 position;
    float2 velocity;
    float radius;
} Ball;

// Helper function to check and resolve collision between two balls
bool resolveCollision(Ball* ball1, Ball* ball2) {
    float2 diff = ball1->position - ball2->position;
    float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
    float minDistance = ball1->radius + ball2->radius;
    
    // Check if balls are colliding
    if (distance < minDistance) {
        // Normal vector of collision
        float2 normal = diff / distance;
        
        // Relative velocity
        float2 relativeVel = ball1->velocity - ball2->velocity;
        
        // Relative velocity along normal
        float velAlongNormal = dot(relativeVel, normal);
        
        // If balls are moving apart, skip collision response
        if (velAlongNormal > 0) return false;
        
        // Simple elastic collision response
        float restitution = 0.8f; // Coefficient of restitution
        float impulse = -(1.0f + restitution) * velAlongNormal;
        
        // Update velocities
        ball1->velocity += normal * impulse;
        ball2->velocity -= normal * impulse;
        
        // Separate balls to prevent sticking
        float correction = (minDistance - distance) * 0.5f;
        ball1->position += normal * correction;
        ball2->position -= normal * correction;
        
        return true;
    }
    return false;
}

// CPU kernel for ball-to-ball collision detection and resolution
__kernel void checkBallCollisions(
    __global Ball* balls,
    const int numBalls,
    __global int* collisionCount  // For statistics/debugging
) {
    int idx = get_global_id(0);
    if (idx >= numBalls) return;
    
    // Each work item handles collisions for one ball against subsequent balls
    Ball localBall = balls[idx];
    int localCollisions = 0;
    
    for (int j = idx + 1; j < numBalls; j++) {
        Ball otherBall = balls[j];
        
        if (resolveCollision(&localBall, &otherBall)) {
            // Update both balls in global memory if collision occurred
            balls[idx] = localBall;
            balls[j] = otherBall;
            localCollisions++;
        }
    }
    
    // Update collision count
    atomic_add(collisionCount, localCollisions);
}

// Optional: CPU kernel for simulation control and statistics
__kernel void updateSimulationStats(
    __global const int* collisionCount,
    __global float* simulationStats,  // Array for various statistics
    const float deltaTime
) {
    // Only one work item needs to run this
    if (get_global_id(0) != 0) return;
    
    // Update various simulation statistics
    simulationStats[0] = (float)*collisionCount;  // Collisions per frame
    simulationStats[1] += deltaTime;              // Total simulation time
    // Add more statistics as needed
}
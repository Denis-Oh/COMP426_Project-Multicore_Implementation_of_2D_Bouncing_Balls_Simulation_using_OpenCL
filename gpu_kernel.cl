#include "ball_def.h"

// Kernel for parallel position updates and wall collision detection
// Simulates GPU-side data-parallel computation on M1 architecture
__kernel void updateBallPositions(
    __global Ball* balls,        // Array of all balls in simulation
    const float deltaTime,       // Time step for physics update
    const FLOAT2 boundaries,     // Window boundaries (width, height)
    const int numBalls          // Total number of balls
) {
    // Get this thread's ball index
    int gid = get_global_id(0);
    if (gid >= numBalls) return;
    
    // Load ball data into local memory for faster access
    Ball ball = balls[gid];
    float radius = ball.radius;
    
    // Apply simplified gravity force (50 units/sec²)
    ball.velocity.y += 50.0f * deltaTime;
    
    // Update position using current velocity
    ball.position.x += ball.velocity.x * deltaTime;
    ball.position.y += ball.velocity.y * deltaTime;
    
    // Wall collision response with energy loss factor
    const float dampening = 0.7f;  // 30% energy loss on collision
    
    // Check and respond to wall collisions
    // Right wall collision
    if (ball.position.x + radius > boundaries.x) {
        ball.position.x = boundaries.x - radius;
        ball.velocity.x = -fabs(ball.velocity.x) * dampening;
    }
    // Left wall collision
    if (ball.position.x - radius < 0) {
        ball.position.x = radius;
        ball.velocity.x = fabs(ball.velocity.x) * dampening;
    }
    
    // Bottom wall collision
    if (ball.position.y + radius > boundaries.y) {
        ball.position.y = boundaries.y - radius;
        ball.velocity.y = -fabs(ball.velocity.y) * dampening;
    }
    // Top wall collision
    if (ball.position.y - radius < 0) {
        ball.position.y = radius;
        ball.velocity.y = fabs(ball.velocity.y) * dampening;
    }
    
    // Apply ground friction when ball is near bottom
    if (fabs(ball.position.y - (boundaries.y - radius)) < 1.0f) {
        ball.velocity.x *= 0.99f;  // 1% velocity loss per frame
    }
    
    // Limit maximum ball speed for stability
    const float MAX_SPEED = 500.0f;
    float speed = sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
    if (speed > MAX_SPEED) {
        float scale = MAX_SPEED / speed;
        ball.velocity.x *= scale;
        ball.velocity.y *= scale;
    }
    
    // Write updated ball data back to global memory
    balls[gid] = ball;
}
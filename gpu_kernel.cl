#include "ball_def.h"

// Helper functions for float2 operations
FLOAT2 float2_add(FLOAT2 a, FLOAT2 b) {
    FLOAT2 result = {a.x + b.x, a.y + b.y};
    return result;
}

FLOAT2 float2_multiply(FLOAT2 a, float scalar) {
    FLOAT2 result = {a.x * scalar, a.y * scalar};
    return result;
}

// Kernel to update ball positions and handle wall collisions
__kernel void updateBallPositions(
    __global Ball* balls,
    const float deltaTime,
    const FLOAT2 boundaries,
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;

    Ball ball = balls[gid];
    float originalRadius = ball.radius;  // Store original radius
    
    // Very light gravity
    ball.velocity.y += 10.0f * deltaTime;
    
    // Apply velocity with dampening if speed is too high
    float maxSpeed = 100.0f;
    float currentSpeed = sqrt(ball.velocity.x * ball.velocity.x + 
                            ball.velocity.y * ball.velocity.y);
    if (currentSpeed > maxSpeed) {
        float scale = maxSpeed / currentSpeed;
        ball.velocity.x *= scale;
        ball.velocity.y *= scale;
    }
    
    // Update position
    ball.position.x += ball.velocity.x * deltaTime;
    ball.position.y += ball.velocity.y * deltaTime;
    
    float dampening = 0.6f;
    
    // Strict boundary enforcement
    // Right wall
    if (ball.position.x + originalRadius >= boundaries.x) {
        ball.position.x = boundaries.x - originalRadius;
        ball.velocity.x = -fabs(ball.velocity.x) * dampening;
    }
    // Left wall
    if (ball.position.x - originalRadius <= 0) {
        ball.position.x = originalRadius;
        ball.velocity.x = fabs(ball.velocity.x) * dampening;
    }
    
    // Bottom wall
    if (ball.position.y + originalRadius >= boundaries.y) {
        ball.position.y = boundaries.y - originalRadius;
        ball.velocity.y = -fabs(ball.velocity.y) * dampening;
    }
    // Top wall
    if (ball.position.y - originalRadius <= 0) {
        ball.position.y = originalRadius;
        ball.velocity.y = fabs(ball.velocity.y) * dampening;
    }
    
    // Apply mild friction to gradually slow down
    float friction = 0.98f;
    ball.velocity.x *= friction;
    ball.velocity.y *= friction;
    
    // Stop very slow movement
    float minSpeed = 0.1f;
    if (currentSpeed < minSpeed) {
        ball.velocity.x = 0.0f;
        ball.velocity.y = 0.0f;
    }
    
    // Ensure radius hasn't changed
    ball.radius = originalRadius;
    
    // Final position verification
    ball.position.x = fmax(originalRadius, fmin(boundaries.x - originalRadius, ball.position.x));
    ball.position.y = fmax(originalRadius, fmin(boundaries.y - originalRadius, ball.position.y));
    
    balls[gid] = ball;
}
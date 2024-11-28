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
    const FLOAT2 boundaries,  // window width and height
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;

    // Load ball data
    Ball ball = balls[gid];
    
    // Update position
    FLOAT2 velocity_delta = float2_multiply(ball.velocity, deltaTime);
    FLOAT2 newPosition = float2_add(ball.position, velocity_delta);
    
    // Handle wall collisions
    // Right and left walls
    if (newPosition.x + ball.radius > boundaries.x) {
        newPosition.x = boundaries.x - ball.radius;
        ball.velocity.x = -ball.velocity.x;
    } else if (newPosition.x - ball.radius < 0) {
        newPosition.x = ball.radius;
        ball.velocity.x = -ball.velocity.x;
    }
    
    // Top and bottom walls
    if (newPosition.y + ball.radius > boundaries.y) {
        newPosition.y = boundaries.y - ball.radius;
        ball.velocity.y = -ball.velocity.y;
    } else if (newPosition.y - ball.radius < 0) {
        newPosition.y = ball.radius;
        ball.velocity.y = -ball.velocity.y;
    }
    
    // Update ball data
    ball.position = newPosition;
    balls[gid] = ball;
}
#include "ball_def.h"

__kernel void updateBallPositions(
    __global Ball* balls,
    const float deltaTime,
    const FLOAT2 boundaries,
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;
    
    Ball ball = balls[gid];
    // Use the ball's own radius instead of constant
    float radius = ball.radius;
    
    // Gravity
    ball.velocity.y += 50.0f * deltaTime;
    
    // Update position
    ball.position.x += ball.velocity.x * deltaTime;
    ball.position.y += ball.velocity.y * deltaTime;
    
    // Boundary collisions with dampening
    const float dampening = 0.7f;
    
    // Right wall
    if (ball.position.x + radius > boundaries.x) {
        ball.position.x = boundaries.x - radius;
        ball.velocity.x = -fabs(ball.velocity.x) * dampening;
    }
    // Left wall
    if (ball.position.x - radius < 0) {
        ball.position.x = radius;
        ball.velocity.x = fabs(ball.velocity.x) * dampening;
    }
    
    // Bottom wall
    if (ball.position.y + radius > boundaries.y) {
        ball.position.y = boundaries.y - radius;
        ball.velocity.y = -fabs(ball.velocity.y) * dampening;
    }
    // Top wall
    if (ball.position.y - radius < 0) {
        ball.position.y = radius;
        ball.velocity.y = fabs(ball.velocity.y) * dampening;
    }
    
    // Apply friction when ball is on ground
    if (fabs(ball.position.y - (boundaries.y - radius)) < 1.0f) {
        ball.velocity.x *= 0.99f;
    }
    
    // Enforce maximum velocity
    const float MAX_SPEED = 500.0f;
    float speed = sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
    if (speed > MAX_SPEED) {
        float scale = MAX_SPEED / speed;
        ball.velocity.x *= scale;
        ball.velocity.y *= scale;
    }
    
    balls[gid] = ball;
}
#include "ball_def.h"

__kernel void updateBallPositions(
    __global Ball* balls,
    const float deltaTime,
    const FLOAT2 boundaries,
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;

    // Use a single constant radius for all balls
    const float BALL_RADIUS = 20.0f;
    
    Ball ball = balls[gid];
    ball.radius = BALL_RADIUS;  // Enforce constant radius
    
    // Gravity (reduced strength)
    ball.velocity.y += 1.0f * deltaTime;
    
    // Update position
    ball.position.x += ball.velocity.x * deltaTime;
    ball.position.y += ball.velocity.y * deltaTime;
    
    // Boundary collisions with dampening
    const float dampening = 0.7f;
    
    // Right wall
    if (ball.position.x + BALL_RADIUS > boundaries.x) {
        ball.position.x = boundaries.x - BALL_RADIUS;
        ball.velocity.x = -fabs(ball.velocity.x) * dampening;
    }
    // Left wall
    if (ball.position.x - BALL_RADIUS < 0) {
        ball.position.x = BALL_RADIUS;
        ball.velocity.x = fabs(ball.velocity.x) * dampening;
    }
    
    // Bottom wall
    if (ball.position.y + BALL_RADIUS > boundaries.y) {
        ball.position.y = boundaries.y - BALL_RADIUS;
        ball.velocity.y = -fabs(ball.velocity.y) * dampening;
    }
    // Top wall
    if (ball.position.y - BALL_RADIUS < 0) {
        ball.position.y = BALL_RADIUS;
        ball.velocity.y = fabs(ball.velocity.y) * dampening;
    }
    
    // Apply friction when ball is on ground
    if (fabs(ball.position.y - (boundaries.y - BALL_RADIUS)) < 1.0f) {
        ball.velocity.x *= 0.99f;
    }
    
    // Enforce maximum velocity
    const float MAX_SPEED = 300.0f;
    float speed = sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
    if (speed > MAX_SPEED) {
        float scale = MAX_SPEED / speed;
        ball.velocity.x *= scale;
        ball.velocity.y *= scale;
    }
    
    balls[gid] = ball;
}
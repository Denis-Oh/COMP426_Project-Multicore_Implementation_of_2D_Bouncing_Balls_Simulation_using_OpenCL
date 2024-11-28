#include "ball_def.h"

// Helper functions for float2 operations
FLOAT2 float2_add(FLOAT2 a, FLOAT2 b) {
    FLOAT2 result = {a.x + b.x, a.y + b.y};
    return result;
}

FLOAT2 float2_subtract(FLOAT2 a, FLOAT2 b) {
    FLOAT2 result = {a.x - b.x, a.y - b.y};
    return result;
}

FLOAT2 float2_multiply_scalar(FLOAT2 a, float scalar) {
    FLOAT2 result = {a.x * scalar, a.y * scalar};
    return result;
}

float float2_dot(FLOAT2 a, FLOAT2 b) {
    return a.x * b.x + a.y * b.y;
}

FLOAT2 float2_divide_scalar(FLOAT2 a, float scalar) {
    FLOAT2 result = {a.x / scalar, a.y / scalar};
    return result;
}

// Helper function to check and resolve collision between two balls
bool resolveCollision(Ball* ball1, Ball* ball2) {
    FLOAT2 diff = float2_subtract(ball1->position, ball2->position);
    float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
    float minDistance = ball1->radius + ball2->radius;
    
    if (distance < minDistance) {
        // Normal vector of collision
        FLOAT2 normal = float2_divide_scalar(diff, distance);
        
        // Relative velocity
        FLOAT2 relativeVel = float2_subtract(ball1->velocity, ball2->velocity);
        
        // Relative velocity along normal
        float velAlongNormal = float2_dot(relativeVel, normal);
        
        // If balls are moving apart, skip collision response
        if (velAlongNormal > 0) return false;
        
        // Simple elastic collision response
        float restitution = 0.8f;
        float impulse = -(1.0f + restitution) * velAlongNormal;
        
        // Update velocities
        FLOAT2 impulseVec = float2_multiply_scalar(normal, impulse);
        ball1->velocity = float2_add(ball1->velocity, impulseVec);
        ball2->velocity = float2_subtract(ball2->velocity, impulseVec);
        
        // Separate balls to prevent sticking
        float correction = (minDistance - distance) * 0.5f;
        FLOAT2 correctionVec = float2_multiply_scalar(normal, correction);
        ball1->position = float2_add(ball1->position, correctionVec);
        ball2->position = float2_subtract(ball2->position, correctionVec);
        
        return true;
    }
    return false;
}

// CPU kernel for ball-to-ball collision detection and resolution
__kernel void checkBallCollisions(
    __global Ball* balls,
    const int numBalls,
    __global int* collisionCount
) {
    int idx = get_global_id(0);
    if (idx >= numBalls) return;
    
    Ball localBall = balls[idx];
    int localCollisions = 0;
    
    for (int j = idx + 1; j < numBalls; j++) {
        Ball otherBall = balls[j];
        
        if (resolveCollision(&localBall, &otherBall)) {
            balls[idx] = localBall;
            balls[j] = otherBall;
            localCollisions++;
        }
    }
    
    atomic_add(collisionCount, localCollisions);
}
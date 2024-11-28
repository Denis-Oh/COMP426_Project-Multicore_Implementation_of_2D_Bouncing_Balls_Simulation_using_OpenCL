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

FLOAT2 float2_normalize(FLOAT2 v) {
    float length = sqrt(v.x * v.x + v.y * v.y);
    if (length < 1e-6f) {
        return (FLOAT2){0.0f, 0.0f};
    }
    return float2_multiply_scalar(v, 1.0f / length);
}

// Helper function to check and resolve collision between two balls
bool resolveCollision(Ball* ball1, Ball* ball2) {
    FLOAT2 diff = float2_subtract(ball1->position, ball2->position);
    float distance = sqrt(float2_dot(diff, diff));
    float minDistance = ball1->radius + ball2->radius;
    
    if (distance < minDistance && distance > 0.0f) {
        // Normalize the difference vector
        FLOAT2 normal = float2_multiply_scalar(diff, 1.0f / distance);
        
        // Calculate relative velocity
        FLOAT2 relativeVel = float2_subtract(ball1->velocity, ball2->velocity);
        
        // Calculate relative velocity along normal
        float velAlongNormal = float2_dot(relativeVel, normal);
        
        // If balls are moving apart, skip collision response
        if (velAlongNormal > 0) return false;
        
        // Calculate collision response
        float restitution = 0.8f;  // Coefficient of restitution
        
        // Calculate impulse scalar
        float j = -(1.0f + restitution) * velAlongNormal;
        j /= 1.0f / ball1->radius + 1.0f / ball2->radius;  // Using radius as mass approximation
        
        // Apply impulse
        FLOAT2 impulse = float2_multiply_scalar(normal, j);
        ball1->velocity = float2_add(ball1->velocity, 
            float2_multiply_scalar(impulse, 1.0f / ball1->radius));
        ball2->velocity = float2_subtract(ball2->velocity, 
            float2_multiply_scalar(impulse, 1.0f / ball2->radius));
        
        // Separate balls to prevent sticking
        float overlap = minDistance - distance;
        FLOAT2 separation = float2_multiply_scalar(normal, overlap * 0.5f);
        ball1->position = float2_add(ball1->position, separation);
        ball2->position = float2_subtract(ball2->position, separation);
        
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
    
    // Check collisions with all subsequent balls
    for (int j = idx + 1; j < numBalls; j++) {
        Ball otherBall = balls[j];
        
        if (resolveCollision(&localBall, &otherBall)) {
            // Write back updated balls
            balls[idx] = localBall;
            balls[j] = otherBall;
            localCollisions++;
        }
    }
    
    atomic_add(collisionCount, localCollisions);
}
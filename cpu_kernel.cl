#include "ball_def.h"

__kernel void checkBallCollisions(
    __global Ball* balls,
    const int numBalls,
    __global int* collisionCount
) {
    int gid = get_global_id(0);
    if (gid >= numBalls - 1) return;
    
    const float BALL_RADIUS = 20.0f;
    Ball ball1 = balls[gid];
    ball1.radius = BALL_RADIUS;  // Ensure constant radius
    
    for (int i = gid + 1; i < numBalls; i++) {
        Ball ball2 = balls[i];
        ball2.radius = BALL_RADIUS;  // Ensure constant radius
        
        // Calculate distance between balls
        float dx = ball2.position.x - ball1.position.x;
        float dy = ball2.position.y - ball1.position.y;
        float distance = sqrt(dx * dx + dy * dy);
        
        // Check for collision
        float minDist = BALL_RADIUS * 2.0f;
        if (distance < minDist && distance > 0.0f) {
            // Normalize collision vector
            float nx = dx / distance;
            float ny = dy / distance;
            
            // Relative velocity
            float dvx = ball2.velocity.x - ball1.velocity.x;
            float dvy = ball2.velocity.y - ball1.velocity.y;
            
            // Relative velocity along normal
            float relativeVelocity = dvx * nx + dvy * ny;
            
            // Don't process if balls are moving apart
            if (relativeVelocity < 0) {
                // Collision response
                float restitution = 0.8f;
                float j = -(1.0f + restitution) * relativeVelocity;
                
                // Apply impulse
                float impulsex = j * nx;
                float impulsey = j * ny;
                
                ball1.velocity.x -= impulsex;
                ball1.velocity.y -= impulsey;
                ball2.velocity.x += impulsex;
                ball2.velocity.y += impulsey;
                
                // Separate the balls
                float overlap = minDist - distance;
                float separationx = nx * overlap * 0.5f;
                float separationy = ny * overlap * 0.5f;
                
                ball1.position.x -= separationx;
                ball1.position.y -= separationy;
                ball2.position.x += separationx;
                ball2.position.y += separationy;
                
                // Write back changes
                balls[gid] = ball1;
                balls[i] = ball2;
                
                atomic_add(collisionCount, 1);
            }
        }
    }
}
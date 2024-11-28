#include "ball_def.h"

__kernel void checkBallCollisions(
    __global Ball* balls,
    const int numBalls,
    __global int* collisionCount
) {
    int gid = get_global_id(0);
    if (gid >= numBalls - 1) return;
    
    Ball ball1 = balls[gid];
    
    for (int i = gid + 1; i < numBalls; i++) {
        Ball ball2 = balls[i];
        
        // Calculate distance between balls
        float dx = ball2.position.x - ball1.position.x;
        float dy = ball2.position.y - ball1.position.y;
        float distance = sqrt(dx * dx + dy * dy);
        
        // Check for collision using actual radii
        float minDist = ball1.radius + ball2.radius;
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
                // Collision response with energy loss
                float restitution = 0.7f;  // Reduced from 0.8 for more energy loss
                
                // Calculate reduced impulse based on relative masses (using radius as mass approximation)
                float mass1 = ball1.radius * ball1.radius;  // Mass proportional to area
                float mass2 = ball2.radius * ball2.radius;
                float totalMass = mass1 + mass2;
                
                float j = -(1.0f + restitution) * relativeVelocity * (mass1 * mass2 / totalMass);
                
                // Apply impulse with mass factoring
                float impulse_factor1 = 1.0f / mass1;
                float impulse_factor2 = 1.0f / mass2;
                
                float impulsex = j * nx;
                float impulsey = j * ny;
                
                // Update velocities based on mass
                ball1.velocity.x -= impulsex * impulse_factor1;
                ball1.velocity.y -= impulsey * impulse_factor1;
                ball2.velocity.x += impulsex * impulse_factor2;
                ball2.velocity.y += impulsey * impulse_factor2;
                
                // Add some energy loss through friction
                ball1.velocity.x *= 0.98f;
                ball1.velocity.y *= 0.98f;
                ball2.velocity.x *= 0.98f;
                ball2.velocity.y *= 0.98f;
                
                // Separate the balls to prevent sticking
                float overlap = minDist - distance;
                float percent = 0.8f;  // Penetration resolution percentage
                float separationx = nx * overlap * percent;
                float separationy = ny * overlap * percent;
                
                // Separate proportionally to mass
                float sep_factor1 = mass2 / totalMass;
                float sep_factor2 = mass1 / totalMass;
                
                ball1.position.x -= separationx * sep_factor1;
                ball1.position.y -= separationy * sep_factor1;
                ball2.position.x += separationx * sep_factor2;
                ball2.position.y += separationy * sep_factor2;
                
                // Write back changes
                balls[gid] = ball1;
                balls[i] = ball2;
                
                atomic_add(collisionCount, 1);
            }
        }
    }
}
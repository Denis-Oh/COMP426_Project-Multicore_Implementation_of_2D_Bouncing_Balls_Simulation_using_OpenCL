#include "ball_def.h"

// Kernel for ball-to-ball collision detection and response
// Simulates CPU-side task parallelism on M1 architecture
__kernel void checkBallCollisions(
    __global Ball* balls,           // Array of all balls in simulation
    const int numBalls,             // Total number of balls
    __global int* collisionCount    // Counter for collisions this frame
) {
    // Get this thread's ball index
    int gid = get_global_id(0);
    if (gid >= numBalls - 1) return;
    
    // Load first ball for comparison
    Ball ball1 = balls[gid];
    
    // Check against all subsequent balls for collisions
    for (int i = gid + 1; i < numBalls; i++) {
        Ball ball2 = balls[i];
        
        // Calculate center-to-center vector between balls
        float dx = ball2.position.x - ball1.position.x;
        float dy = ball2.position.y - ball1.position.y;
        float distance = sqrt(dx * dx + dy * dy);
        
        // Detect collision using combined radii
        float minDist = ball1.radius + ball2.radius;
        if (distance < minDist && distance > 0.0f) {
            // Calculate normalized collision normal
            float nx = dx / distance;
            float ny = dy / distance;
            
            // Calculate relative velocity vector
            float dvx = ball2.velocity.x - ball1.velocity.x;
            float dvy = ball2.velocity.y - ball1.velocity.y;
            
            // Project relative velocity onto collision normal
            float relativeVelocity = dvx * nx + dvy * ny;
            
            // Only process collision if balls are moving toward each other
            if (relativeVelocity < 0) {
                // Collision elasticity (30% energy loss)
                float restitution = 0.7f;
                
                // Calculate mass based on ball area
                float mass1 = ball1.radius * ball1.radius;
                float mass2 = ball2.radius * ball2.radius;
                float totalMass = mass1 + mass2;
                
                // Calculate collision impulse magnitude
                float j = -(1.0f + restitution) * relativeVelocity * (mass1 * mass2 / totalMass);
                
                // Calculate impulse factors based on mass
                float impulse_factor1 = 1.0f / mass1;
                float impulse_factor2 = 1.0f / mass2;
                
                // Convert impulse to vector components
                float impulsex = j * nx;
                float impulsey = j * ny;
                
                // Apply impulses proportional to mass
                ball1.velocity.x -= impulsex * impulse_factor1;
                ball1.velocity.y -= impulsey * impulse_factor1;
                ball2.velocity.x += impulsex * impulse_factor2;
                ball2.velocity.y += impulsey * impulse_factor2;
                
                // Apply collision friction (2% energy loss)
                ball1.velocity.x *= 0.98f;
                ball1.velocity.y *= 0.98f;
                ball2.velocity.x *= 0.98f;
                ball2.velocity.y *= 0.98f;
                
                // Resolve ball overlap to prevent sticking
                float overlap = minDist - distance;
                float percent = 0.8f;  // Resolve 80% of overlap
                float separationx = nx * overlap * percent;
                float separationy = ny * overlap * percent;
                
                // Separate balls proportional to their masses
                float sep_factor1 = mass2 / totalMass;
                float sep_factor2 = mass1 / totalMass;
                
                ball1.position.x -= separationx * sep_factor1;
                ball1.position.y -= separationy * sep_factor1;
                ball2.position.x += separationx * sep_factor2;
                ball2.position.y += separationy * sep_factor2;
                
                // Update ball states in global memory
                balls[gid] = ball1;
                balls[i] = ball2;
                
                // Increment collision counter atomically
                atomic_add(collisionCount, 1);
            }
        }
    }
}
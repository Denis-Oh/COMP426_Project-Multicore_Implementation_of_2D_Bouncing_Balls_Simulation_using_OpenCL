#define THREADS_PER_BALL 32
#define LOCAL_SIZE 256

__kernel void updateBalls(
    __global float4* positions,
    __global float4* velocities,
    __global float4* colors,
    const int numBalls,
    const float gravity,
    const float windowWidth,
    const float windowHeight
) {
    __local float4 local_pos[16];
    __local float4 local_vel[16];
    __local float4 collision_impacts[16];
    __local int collision_counts[16];  // Track number of collisions per ball
    
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int ball_idx = global_id / THREADS_PER_BALL;
    int thread_idx = global_id % THREADS_PER_BALL;
    
    if (ball_idx >= numBalls) return;
    
    // Initialize local memory
    if (thread_idx == 0) {
        collision_impacts[ball_idx] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        collision_counts[ball_idx] = 0;
        local_pos[ball_idx] = positions[ball_idx];
        local_vel[ball_idx] = velocities[ball_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float4 pos = local_pos[ball_idx];
    float4 vel = local_vel[ball_idx];
    
    // Phase 1: Update positions (single thread per ball)
    if (thread_idx == 0) {
        vel.y += gravity;
        pos.x += vel.x;
        pos.y += vel.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Phase 2: Boundary collisions (single thread per ball)
    if (thread_idx == 0) {
        bool collided = false;
        float dampening = 0.95f;
        
        // X boundaries
        if (pos.x - pos.z < 0) {
            pos.x = pos.z;
            vel.x = -vel.x * dampening;
            collided = true;
        } else if (pos.x + pos.z > windowWidth) {
            pos.x = windowWidth - pos.z;
            vel.x = -vel.x * dampening;
            collided = true;
        }
        
        // Y boundaries
        if (pos.y - pos.z < 0) {
            pos.y = pos.z;
            vel.y = -vel.y * dampening;
            collided = true;
        } else if (pos.y + pos.z > windowHeight) {
            pos.y = windowHeight - pos.z;
            vel.y = -vel.y * dampening;
            collided = true;
        }
        
        local_pos[ball_idx] = pos;
        local_vel[ball_idx] = vel;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Phase 3: Ball-to-ball collisions
    if (thread_idx < numBalls && thread_idx != ball_idx) {
        float4 other_pos = local_pos[thread_idx];
        float4 other_vel = local_vel[thread_idx];
        
        float dx = other_pos.x - pos.x;
        float dy = other_pos.y - pos.y;
        float distance = sqrt(dx * dx + dy * dy);
        float minDist = pos.z + other_pos.z;
        
        if (distance < minDist && distance > 0.0f) {
            // Normalized collision vector
            float nx = dx / distance;
            float ny = dy / distance;
            
            // Separate the balls
            float overlap = minDist - distance;
            if (ball_idx < thread_idx) {
                pos.x -= nx * overlap * 0.5f;
                pos.y -= ny * overlap * 0.5f;
            }
            
            // Relative velocity
            float rvx = other_vel.x - vel.x;
            float rvy = other_vel.y - vel.y;
            float velAlongNormal = rvx * nx + rvy * ny;
            
            // Only resolve if objects are moving toward each other
            if (velAlongNormal < 0) {
                float restitution = 0.8f;
                float j = -(1.0f + restitution) * velAlongNormal;
                
                // Store collision impact
                float4 impact = collision_impacts[ball_idx];
                impact.x += -j * nx;
                impact.y += -j * ny;
                collision_impacts[ball_idx] = impact;
                
                // Increment collision count
                atomic_inc(&collision_counts[ball_idx]);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Phase 4: Apply collision impacts and write back results (single thread per ball)
    if (thread_idx == 0) {
        // Apply accumulated collision impacts with scaling
        float4 impact = collision_impacts[ball_idx];
        int count = collision_counts[ball_idx];
        
        if (count > 0) {
            // Scale the impact by number of collisions to prevent energy gain
            vel.x += impact.x / (float)count;
            vel.y += impact.y / (float)count;
        }
        
        // Apply velocity limits
        float maxSpeed = 5.0f;  // Reduced max speed
        float speedSq = vel.x * vel.x + vel.y * vel.y;
        if (speedSq > maxSpeed * maxSpeed) {
            float scale = maxSpeed / sqrt(speedSq);
            vel.x *= scale;
            vel.y *= scale;
        }
        
        positions[ball_idx] = pos;
        velocities[ball_idx] = vel;
    }
}
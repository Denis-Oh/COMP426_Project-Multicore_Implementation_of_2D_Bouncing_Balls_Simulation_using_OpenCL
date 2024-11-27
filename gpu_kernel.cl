// Structure to represent a ball
typedef struct {
    float2 position;     // x, y position
    float2 velocity;     // x, y velocity components
    float radius;        // ball radius
} Ball;

// Kernel to update ball positions and handle wall collisions
__kernel void updateBallPositions(
    __global Ball* balls,
    const float deltaTime,
    const float2 boundaries,  // window width and height
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;

    // Load ball data
    Ball ball = balls[gid];
    
    // Update position
    float2 newPosition = ball.position + ball.velocity * deltaTime;
    
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

// Kernel to prepare visualization data for OpenGL
__kernel void prepareVisualization(
    __global const Ball* balls,
    __global float4* vertices,  // x, y, radius, unused
    const int numBalls
) {
    int gid = get_global_id(0);
    if (gid >= numBalls) return;
    
    Ball ball = balls[gid];
    vertices[gid] = (float4)(ball.position.x, ball.position.y, ball.radius, 0.0f);
}
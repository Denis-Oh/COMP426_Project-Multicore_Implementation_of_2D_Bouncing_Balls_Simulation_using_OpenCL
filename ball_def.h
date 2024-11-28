#ifndef BALL_DEF_H
#define BALL_DEF_H

#ifdef __OPENCL_VERSION__
    // Use OpenCL's built-in float2 when compiling for OpenCL
    #define FLOAT2 float2
#else
    // Define float2 struct for CPU code with both array and x/y access
    typedef union {
        struct {
            float x;
            float y;
        };
        float s[2];
    } FLOAT2;
#endif

typedef struct {
    FLOAT2 position;    // x, y position
    FLOAT2 velocity;    // x, y velocity components
    float radius;       // ball radius
} Ball;

#endif // BALL_DEF_H
#ifndef BALL_DEF_H
#define BALL_DEF_H

#ifdef __OPENCL_VERSION__
    #define FLOAT2 float2
#else
    typedef struct {
        float x;
        float y;
    } __attribute__((aligned(8))) FLOAT2;  // Ensure 8-byte alignment
#endif

typedef struct {
    FLOAT2 position;    // 8 bytes
    FLOAT2 velocity;    // 8 bytes
    float radius;       // 4 bytes
    float padding;      // 4 bytes for alignment
} __attribute__((aligned(16))) Ball;  // Ensure 16-byte alignment

#endif // BALL_DEF_H
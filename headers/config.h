#ifndef CONFIG_H
#define CONFIG_H

// 🔢 Buffer settings
#define BUFFER_NUMBER 4

// Uncomment to enable features
//#define RANDOM_SEED 123456789  
//#define IMG_PATH "image.png"
//#define TIMEUP_fUNC
//#define RECORD_CSV
//#define STEP_BY_STEP_MOD

// 🌐 Grid parameters
#define GRID_SIZE 1024        // Must be a power of 2
#define RADIUS    (13*9)      // Neighborhood radius
#define DELTA_T   0.01

//#define REGION_SIZE 256       // Must be greater than 2*RADIUS + 1

// Derived constants
#ifndef REGION_SIZE
    #define FFT_SIZE_REG GRID_SIZE
#else
    #define SUB_REG_SIZE       (REGION_SIZE / 16)
    #define REGION_NUMBER      (GRID_SIZE / REGION_SIZE)
    #define SUB_REG_NUMBER     (GRID_SIZE / SUB_REG_SIZE)
    #define SUB_REG_PER_REG    (REGION_SIZE / SUB_REG_SIZE)
    #define FFT_SIZE_REG       (REGION_SIZE * 2)

    #define DEBUG_LIFE_MOD
#endif

// ⏱ Timing macros
#ifdef TIMEUP_fUNC

#define CUDA_TIMING_BEGIN(label) \
    cudaEvent_t label##_start, label##_end; \
    cudaEventCreate(&label##_start); \
    cudaEventCreate(&label##_end); \
    cudaEventRecord(label##_start,0)

#define CUDA_TIMING_END(label) \
    cudaEventRecord(label##_end,0); \
    cudaEventSynchronize(label##_end); \
    float label##_ms = 0; \
    cudaEventElapsedTime(&label##_ms, label##_start, label##_end); \
    printf("[GPU] " #label " took %.3f ms\n", label##_ms); \
    cudaEventDestroy(label##_start); \
    cudaEventDestroy(label##_end)

#define CPU_TIMING_BEGIN(label) clock_t label##_start = clock();
#define CPU_TIMING_END(label) \
    clock_t label##_end = clock(); \
    printf(#label " time: %.3f ms\n", 1000.0 * (label##_end - label##_start) / CLOCKS_PER_SEC);

#else
#define CUDA_TIMING_BEGIN(label) 
#define CUDA_TIMING_END(label)
#define CPU_TIMING_BEGIN(label)
#define CPU_TIMING_END(label)
#endif

// 🔐 Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUFFT(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error: %d (%s:%d)\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // CONFIG_H

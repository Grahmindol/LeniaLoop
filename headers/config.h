#ifndef CONFIG_H
#define CONFIG_H

#define BUFFER_NUMBER 2

//#define RANDOM_SEED 123456789  
//#define IMG_PATH "roza.png"
//#define DEBUG_LIFE_MOD
//#define TIMEUP_fUNC
//#define RECORD_CSV
//#define STEP_BY_STEP_MOD
#define DISPLAY

#define GENETIC 3000


// Define grid dimensions using macros
#define GRID_SIZE 128 // MUST BE a POWER OF 2
#define RADIUS 13//(13*9) // 19*9
#define DELTA_T 0.005



#define REGION_SIZE 64 // MUST BE GRATHER THAN 2*RADIUS+1
#define SUB_REG_PER_REG 16


#ifdef REGION_SIZE

#define SUB_REG_SIZE (REGION_SIZE / SUB_REG_PER_REG)
#define REGION_NUMBER (GRID_SIZE/REGION_SIZE)
#define SUB_REG_NUMBER (GRID_SIZE/SUB_REG_SIZE)
#define FFT_SIZE_REG (REGION_SIZE * 2)
#else 
#define FFT_SIZE_REG GRID_SIZE
#endif


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
#define CPU_TIMING_END(label) clock_t label##_end = clock(); \
        printf(#label " time: %.3f ms\n", 1000.0 * (label##_end - label##_start) / CLOCKS_PER_SEC);
#else
#define CUDA_TIMING_BEGIN(label) 
#define CUDA_TIMING_END(label)
#define CPU_TIMING_BEGIN(label)
#define CPU_TIMING_END(label)
#endif



// 🔐 Macros pour vérification d’erreurs
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
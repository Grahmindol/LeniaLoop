#include "life_updater.h"
#include "cuda_convolve_utils.h"
#include "config.h"
#include <cstdio>

// =========================================================
//  GPU Buffers
// =========================================================
static float* d_convolution = nullptr;
static float** d_pixels     = nullptr;
static float* h_pixel_ptrs[BUFFER_NUMBER];
static int* d_roi = nullptr;

// =========================================================
//  Device Helpers
// =========================================================
__device__ static inline float bell(float x, float m, float s) {
    return expf(-powf((x - m) / s, 2) / 2);
}

__device__ static inline float growth(float x) {
    if (x > 0.3f) return -1.0f;
    return 2.0f * bell(x, 0.15f, 0.015f) - 1.0f;
}

__device__ static inline float clip(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}

// =========================================================
//  Main Update Kernel
// =========================================================
__global__ void update_kernel(
    float* d_pixels_current,
    float* d_pixels_next,
    float* d_convolution,
    const int* d_roi
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) return;

    const int i = (y * GRID_SIZE + x) * 3;
    float live_neighbors = d_convolution[y * GRID_SIZE + x];
    float new_state = clip(d_pixels_current[i] + growth(live_neighbors) * DELTA_T);

    d_pixels_next[i] = new_state;

#ifndef DEBUG_LIFE_MOD
    d_pixels_next[i + 2] = live_neighbors;
#else
    int roi_idx = (x / SUB_REG_SIZE) * SUB_REG_NUMBER + (y / SUB_REG_SIZE);

    if (d_roi[roi_idx] & 0b0100)       d_pixels_next[i + 1] = 1.0f;
    else if (d_roi[roi_idx] & 0b1000)  d_pixels_next[i + 1] = 0.5f;
    else                               d_pixels_next[i + 1] = 0.0f;

    if (live_neighbors != 0) {
        d_pixels_next[i + 2] = 0.5f;
        d_convolution[y * GRID_SIZE + x] = 0;
    } else {
        d_pixels_next[i + 2] = 0.0f;
    }
#endif
}

// =========================================================
//  Host Init / Destroy
// =========================================================
void init_updater(float* h_pixels) {
    // Allocate GPU buffers
    cudaMalloc((void**)&d_pixels, sizeof(float*) * BUFFER_NUMBER);

    for (size_t i = 0; i < BUFFER_NUMBER; i++) {
        cudaMalloc((void**)&h_pixel_ptrs[i],
                   sizeof(float) * GRID_SIZE * GRID_SIZE * 3);
    }

    // Copy host pointer array to device
    cudaMemcpy(d_pixels, h_pixel_ptrs,
               sizeof(float*) * BUFFER_NUMBER,
               cudaMemcpyHostToDevice);

    // Copy initial data into buffer[0]
    cudaMemcpy(h_pixel_ptrs[0], h_pixels,
               sizeof(float) * GRID_SIZE * GRID_SIZE * 3,
               cudaMemcpyHostToDevice);

    // Convolution buffer
    cudaMalloc((void**)&d_convolution, sizeof(float) * GRID_SIZE * GRID_SIZE);

#ifdef REGION_SIZE
    cudaMalloc((void**)&d_roi, sizeof(int) * SUB_REG_NUMBER * SUB_REG_NUMBER);
    cudaMemset(d_roi, 1, sizeof(int) * SUB_REG_NUMBER * SUB_REG_NUMBER);
#endif

    CHECK_CUDA(cudaGetLastError());
}

void destroy_updater() {
    for (size_t i = 0; i < BUFFER_NUMBER; i++) {
        cudaFree(h_pixel_ptrs[i]);
    }
    cudaFree(d_pixels);
    cudaFree(d_convolution);

#ifdef REGION_SIZE
    cudaFree(d_roi);
#endif

    CHECK_CUDA(cudaGetLastError());
}

// =========================================================
//  Region Handling (ROI)
// =========================================================
#ifdef REGION_SIZE

__device__ int isActiveSubReg(int sx, int sy, float** datas, int frame) {
    int x_start = sx * SUB_REG_SIZE;
    int y_start = sy * SUB_REG_SIZE;

    for (int x = x_start; x < x_start + SUB_REG_SIZE; ++x) {
        for (int y = y_start; y < y_start + SUB_REG_SIZE; ++y) {
            int i = (y * GRID_SIZE + x) * 3;
            if (datas[frame][i] > 0) return 1;
        }
    }
    return 0;
}

__global__ void update_update_sub_region_interst(
    float** d_pixels,
    int frame,
    int* d_roi
) {
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= SUB_REG_NUMBER || sy >= SUB_REG_NUMBER) return;

    int idx = sx * SUB_REG_NUMBER + sy;
    if (!d_roi[idx]) return;

    d_roi[idx] = isActiveSubReg(sx, sy, d_pixels, frame);
}

__global__ void kernel_expand_sub_region_interst(int* d_roi) {
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= SUB_REG_NUMBER || sy >= SUB_REG_NUMBER) return;

    int idx = sx * SUB_REG_NUMBER + sy;

    if (d_roi[idx] & 1) {
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int nx = (sx + dx + SUB_REG_NUMBER) % SUB_REG_NUMBER;
                int ny = (sy + dy + SUB_REG_NUMBER) % SUB_REG_NUMBER;
                int nidx = nx * SUB_REG_NUMBER + ny;
                atomicOr(&d_roi[nidx], 0b10);
            }
        }
    }
}

__global__ void kernel_propagate_sub_region_interst(int* d_roi, int* d_changed) {
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= SUB_REG_NUMBER || sy >= SUB_REG_NUMBER) return;

    int idx = sx * SUB_REG_NUMBER + sy;
    if (d_roi[idx]) return;

    int idx_right = (sx + 1) * SUB_REG_NUMBER + sy;
    int idx_down  = sx * SUB_REG_NUMBER + (sy + 1);

    if (!d_roi[idx_right] || !d_roi[idx_down]) return;

    atomicOr(&d_roi[idx], 0b10);
    atomicOr(d_changed, 0b10);
}

static inline void setup_sub_region_interst(
    int* d_roi, float** d_datas, int frame
) {
    CPU_TIMING_BEGIN(setupSubRegionInterst);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((SUB_REG_NUMBER + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (SUB_REG_NUMBER + threadsPerBlock.y - 1) / threadsPerBlock.y);

    update_update_sub_region_interst<<<blocksPerGrid, threadsPerBlock>>>(
        d_datas, frame, d_roi
    );
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    kernel_expand_sub_region_interst<<<blocksPerGrid, threadsPerBlock>>>(d_roi);
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    int changed;
    do {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        kernel_propagate_sub_region_interst<<<blocksPerGrid, threadsPerBlock>>>(
            d_roi, d_changed
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (changed);

    cudaFree(d_changed);
    CPU_TIMING_END(setupSubRegionInterst);
}
#endif // REGION_SIZE

// =========================================================
//  Main Update Function
// =========================================================
void update_life(int current, float* h_pixels_result) {
    static int iter = 0;

    // --- Convolution
CUDA_TIMING_BEGIN(convolution_cuda);

    convolve2d_circular(h_pixel_ptrs[current], d_roi, d_convolution, NULL);

CUDA_TIMING_END(convolution_cuda);

    // --- Update Pixels
    CUDA_TIMING_BEGIN(updatePixels);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((GRID_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (GRID_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

    update_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        h_pixel_ptrs[current],
        h_pixel_ptrs[(current + 1) % BUFFER_NUMBER],
        d_convolution,
        d_roi
    );
    cudaDeviceSynchronize();

    // Copy back results occasionally
#ifdef DISPLAY
#ifndef STEP_BY_STEP_MOD
    if (iter > DISPLAY) {
#endif
        cudaMemcpy(h_pixels_result,
                   h_pixel_ptrs[current],
                   sizeof(float) * GRID_SIZE * GRID_SIZE * 3,
                   cudaMemcpyDeviceToHost);
        iter = 0;
#ifndef STEP_BY_STEP_MOD
    }
#endif
#endif

#ifdef REGION_SIZE
    setup_sub_region_interst(d_roi, (float**)d_pixels, (current + 1) % BUFFER_NUMBER);
#endif

    CUDA_TIMING_END(updatePixels);
    iter++;
}

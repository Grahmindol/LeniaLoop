#include "cuda_convolve_utils.h"

#include "cuda_fft_utils.h"
#include "config.h"

#include <stdlib.h>
#include <stdio.h>
//#include "csv_util.h"

// connvolve buffers
static cufftComplex *d_KernelFFT_reg = NULL;
static cufftComplex *d_imageFFT_reg_batchs = NULL;
static cufftComplex *d_resultFFT_reg_batchs = NULL;

#ifdef REGION_SIZE
// batching system buffer
static int *batch_x_dev = NULL, *batch_y_dev = NULL, g_batch_count = 0;
static int *d_new_batches;
#endif
/**
 * @brief Precompute the FFT of a symmetric square kernel for full image and region-based convolution.
 *
 * @param kernel Pointer to float kernel values (row-major order), assumed symmetric
 * @param size   Kernel size (n x n), must satisfy size <= FFT_size_reg
 */
void precompute_kernelFFT(const float *kernel, int size)
{   
    #ifdef REGION_SIZE
    if (size > REGION_SIZE){
        printf("ERROR: region size too small for the kernel...\n");
        exit(1);
    }
    #else
    if (size > GRID_SIZE){
        printf("ERROR: grid size too small for the kernel...\n");
        exit(1);
    }
    #endif

    if (d_KernelFFT_reg){
        cudaFree(d_KernelFFT_reg);
        d_KernelFFT_reg = NULL;
    }
    printf("INFO: pre-calculating kernel FFT of size %dx%d\n", size, size);

    cufftComplex * KernelFFT_reg = (cufftComplex *)malloc(FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex));
    if(!KernelFFT_reg){
        fprintf(stderr,"ERROR: allocating convlove buffers !");
    }
    
    // --- Center kernel in KernelFFT_reg ---
    int startX = (FFT_SIZE_REG - size) / 2;
    int startY = (FFT_SIZE_REG - size) / 2;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (startY + y < FFT_SIZE_REG && startX + x < FFT_SIZE_REG) {
                KernelFFT_reg[((startY + y) * FFT_SIZE_REG) + startX + x] = (cufftComplex){kernel[x + y * size], 0};
            }
        }
    }

    // --- Compute FFTs of both kernel versions ---
    cudaError_t err = cudaMalloc((void**)&d_KernelFFT_reg, FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copier host → device
    err = cudaMemcpy(d_KernelFFT_reg, KernelFFT_reg, FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    init_FFT_plans(1);
    fft2d_batch(d_KernelFFT_reg);
    setup_kenel_texture(d_KernelFFT_reg);
    free(KernelFFT_reg);
}

void init_convolve_buffer(){
    #ifdef REGION_SIZE
    size_t batch_array_size = 2 * REGION_NUMBER * REGION_NUMBER * sizeof(int);
    size_t total_bytes = 2 * REGION_NUMBER * REGION_NUMBER * FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex);
    #else
    size_t total_bytes = FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex);
    #endif
    CHECK_CUDA(cudaMalloc((void**)&d_imageFFT_reg_batchs, total_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_resultFFT_reg_batchs, total_bytes));

    #ifdef REGION_SIZE
    CHECK_CUDA(cudaMalloc((void**)&batch_x_dev, batch_array_size));
    CHECK_CUDA(cudaMalloc((void**)&batch_y_dev, batch_array_size));
    CHECK_CUDA(cudaMalloc((void**)&d_new_batches,  sizeof(int)));
    #endif
}

void destroy_convolve_buffer(){
    #ifdef REGION_SIZE
    // 🧹 Libération
    CHECK_CUDA(cudaFree(batch_x_dev));
    CHECK_CUDA(cudaFree(batch_y_dev));
    CHECK_CUDA(cudaFree(d_new_batches));
    #endif

    destroy_FFT_plans();
    free_kernel_texture();

    if (d_imageFFT_reg_batchs){
        CHECK_CUDA(cudaFree(d_imageFFT_reg_batchs));
        d_imageFFT_reg_batchs = NULL;
    }  

    if (d_resultFFT_reg_batchs){
        CHECK_CUDA(cudaFree(d_resultFFT_reg_batchs));
        d_resultFFT_reg_batchs = NULL;
    } 

    if (d_KernelFFT_reg){
        CHECK_CUDA(cudaFree(d_KernelFFT_reg));
        d_KernelFFT_reg = NULL;
    } 

}

__device__ cufftComplex get_value_device(const float *img, const int x, const int y)
{
    int xi = x & (GRID_SIZE - 1);
    int yi = y & (GRID_SIZE - 1);
    int idx = ((xi * GRID_SIZE) | yi) * 3;
    return (cufftComplex){img[idx], 0};
}

#ifdef REGION_SIZE



__global__ void copy_and_pad_kernel_large(
    cufftComplex *dst_batches,
    const float *image,
    const int *batch_positions_x,
    const int *batch_positions_y,
    int batch_count)
{
    __shared__ int xc, yc;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int region_idx = blockIdx.z;

    if (region_idx >= batch_count || x >= FFT_SIZE_REG || y >= FFT_SIZE_REG)
        return;

    if (tx == 0 && ty == 0) {
        xc = batch_positions_x[region_idx];
        yc = batch_positions_y[region_idx];
    }
    __syncthreads();

    int quarter = FFT_SIZE_REG >> 2;

    int img_x = xc + x - quarter;
    int img_y = yc + y - quarter;

    cufftComplex val = get_value_device(image, img_x, img_y);
    int dst_idx = region_idx * FFT_SIZE_REG * FFT_SIZE_REG + y * FFT_SIZE_REG + x;
    dst_batches[dst_idx] = val;
}

static inline void copy_region_buffer_with_padding(
    cufftComplex* dst_dev,           // buffer host de sortie (device)
    const float* image_dev          // image source (device)
    )
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (FFT_SIZE_REG + threadsPerBlock.y - 1) / threadsPerBlock.y,
              g_batch_count);

    copy_and_pad_kernel_large<<<blocksPerGrid, threadsPerBlock>>>(
        dst_dev,
        image_dev,
        batch_x_dev,
        batch_y_dev,
        g_batch_count
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void remove_padding_and_center_shift_kernel(
    cufftComplex* __restrict__ src_batch,
    float* __restrict__ result_out,
    const int* __restrict__ pos_x,
    const int* __restrict__ pos_y,
    const int batch_count
) {
    int k = blockIdx.z;
    if (k >= batch_count) return;
    int half = FFT_SIZE_REG >> 1;
    int quarter = FFT_SIZE_REG >> 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in N/2
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col in N/2

    if (i >= half || j >= half) return;

    // Compute index within the padded FFT block
    int shift_i = (i + quarter) ^ half;
    int shift_j = (j + quarter) ^ half;

    // Apply FFT shift (XOR swap quadrants)
    int shifted_idx = shift_i * FFT_SIZE_REG + shift_j;
    float val = src_batch[k * FFT_SIZE_REG * FFT_SIZE_REG + shifted_idx].x;

    // Compute wrapped output coordinates
    int x = pos_x[k];
    int y = pos_y[k];
    int out_x = (x + j + GRID_SIZE) % GRID_SIZE;
    int out_y = (y + i + GRID_SIZE) % GRID_SIZE;

    result_out[out_y + out_x * GRID_SIZE] = val;
}


static inline void remove_padding_and_center_shift_cuda(
    cufftComplex* d_resultFFT_reg_batchs,
    float* d_result_out
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((FFT_SIZE_REG / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (FFT_SIZE_REG / 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
              g_batch_count);

    remove_padding_and_center_shift_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_resultFFT_reg_batchs,
        d_result_out,
        batch_x_dev,
        batch_y_dev,
        g_batch_count
    );

    cudaDeviceSynchronize(); // ou à la fin d'une pipeline
}
#define FLAG_ACTIVE     0b0011
#define FLAG_CORNER     0b0100
#define FLAG_MASKED     0b1000

__global__ void detectCornersKernel(int* d_sub_roi,
                                      int* d_batch_x,
                                      int* d_batch_y,
                                      int* d_new_batches)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= SUB_REG_NUMBER || sy >= SUB_REG_NUMBER) return;

    int idx = sy * SUB_REG_NUMBER + sx;
    int v = d_sub_roi[idx] & FLAG_ACTIVE;
    if (!v) return;

    bool left_valid = (sy > 0) && (d_sub_roi[(sy - 1) * SUB_REG_NUMBER + sx] & FLAG_ACTIVE);
    bool up_valid   = (sx > 0) && (d_sub_roi[sy * SUB_REG_NUMBER + (sx - 1)] & FLAG_ACTIVE);
    if (left_valid || up_valid) return;

    int insert = atomicAdd(d_new_batches, 1);
    d_batch_x[insert] = sx * SUB_REG_SIZE;
    d_batch_y[insert] = sy * SUB_REG_SIZE;

    #ifdef DEBUG_LIFE_MOD
    #ifdef STEP_BY_STEP_MOD
    printf("reg n°%d found at (%d,%d)\n", insert - 1, sx, sy);
    #endif
    atomicOr(&d_sub_roi[idx], FLAG_CORNER);
    #endif
}

__global__ void maskRegionsKernel(int* d_sub_roi,
                                    int* d_batch_x,
                                    int* d_batch_y,
                                    int* d_new_batches)
{
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int r_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (r_id >= *d_new_batches) return;
    if (dx >= SUB_REG_PER_REG || dy >= SUB_REG_PER_REG) return;

    int gx = d_batch_x[r_id] / SUB_REG_SIZE + dx;
    int gy = d_batch_y[r_id] / SUB_REG_SIZE + dy;

    if (gx >= SUB_REG_NUMBER || gy >= SUB_REG_NUMBER) return;

    if (d_sub_roi[gy * SUB_REG_NUMBER + gx] & FLAG_ACTIVE){  
        #ifdef DEBUG_LIFE_MOD
        if (d_sub_roi[gy * SUB_REG_NUMBER + gx] & 0b01)
        {
            atomicOr(&d_sub_roi[gy * SUB_REG_NUMBER + gx], FLAG_MASKED);
        }else
        {
            atomicOr(&d_sub_roi[gy * SUB_REG_NUMBER + gx], 0b00010000);
        }
        #else
        atomicOr(&d_sub_roi[gy * SUB_REG_NUMBER + gx], FLAG_MASKED);
        #endif
        atomicAnd(&d_sub_roi[gy * SUB_REG_NUMBER + gx], ~FLAG_ACTIVE);
    }
}


int compute_batch_positions(int* d_sub_roi)
{
    int total_batches = 0;
    int h_new;
    int connex_reg;

 
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((SUB_REG_NUMBER + 31) / 32, (SUB_REG_NUMBER + 31) / 32);

    // reset new counter
    cudaMemset(d_new_batches, 0, sizeof(int));

        // detect corners
    detectCornersKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sub_roi, batch_x_dev+total_batches, batch_y_dev+total_batches, d_new_batches);
    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(&h_new, d_new_batches, sizeof(int), cudaMemcpyDeviceToHost);
    if ((connex_reg = h_new) == 0) return 0;  // 🛑 no more

    // iterative loop
    while (true) {
        // mask regions
        dim3 threadsPerBlock3D(16, 16, 4);

        int blocksX = (SUB_REG_PER_REG + threadsPerBlock3D.x - 1) / threadsPerBlock3D.x;
        int blocksY = (SUB_REG_PER_REG + threadsPerBlock3D.y - 1) / threadsPerBlock3D.y;
        int blocksZ = (h_new + threadsPerBlock3D.z - 1) / threadsPerBlock3D.z;

        dim3 blocksPerGrid3D(blocksX, blocksY, blocksZ);
        maskRegionsKernel<<<blocksPerGrid3D, threadsPerBlock3D>>>(d_sub_roi, batch_x_dev+total_batches, batch_y_dev+total_batches, d_new_batches);
        CHECK_CUDA(cudaGetLastError());

        total_batches += h_new;
        #ifdef STEP_BY_STEP_MOD
        printf("h_new : %d \n", h_new);
        #endif

        // reset new counter
        cudaMemset(d_new_batches, 0, sizeof(int));

        // detect corners
        detectCornersKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sub_roi, batch_x_dev+total_batches, batch_y_dev+total_batches, d_new_batches);
        CHECK_CUDA(cudaGetLastError());

        cudaMemcpy(&h_new, d_new_batches, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_new == 0) break;  // 🛑 no more

    }
    #ifdef STEP_BY_STEP_MOD
    printf("--total_batch : %d \n", total_batches);
    #endif

    g_batch_count = total_batches;
    return connex_reg;
}


int convolve2d_fft_circular(const float* d_image, int* d_roi, float* d_result_out, float* debug) {
    //RECORD_CPU_TIMING_BEGIN(convolve_times)
    //RECORD_CPU_TIMING_BEGIN(batch_positions_times)
    int connex_reg = compute_batch_positions(d_roi);
    static int last_batch = 0;
    if (g_batch_count == 0) 
        return connex_reg;
    if (g_batch_count != last_batch){
        init_FFT_plans(g_batch_count);
        setup_kenel_texture(d_KernelFFT_reg);
        last_batch = g_batch_count;
    }
    //RECORD_CPU_TIMING_END(batch_positions_times)

    //RECORD_CPU_TIMING_BEGIN(copy_times)
    copy_region_buffer_with_padding(d_imageFFT_reg_batchs,d_image);
    //RECORD_CPU_TIMING_END(copy_times)
    //RECORD_CPU_TIMING_BEGIN(FFT_times)
    fft2d_batch(d_imageFFT_reg_batchs);
    //RECORD_CPU_TIMING_END(FFT_times)
    //RECORD_CPU_TIMING_BEGIN(broadcast_times)
    complex_pointwise_broadcast(d_imageFFT_reg_batchs,d_resultFFT_reg_batchs,g_batch_count);
    //RECORD_CPU_TIMING_END(broadcast_times)
    //RECORD_CPU_TIMING_BEGIN(IFFT_times)
    ifft2d_batch(d_resultFFT_reg_batchs);
    //RECORD_CPU_TIMING_END(IFFT_times)
    //RECORD_CPU_TIMING_BEGIN(shift_times)
    remove_padding_and_center_shift_cuda(d_resultFFT_reg_batchs,d_result_out);
    //RECORD_CPU_TIMING_END(shift_times)

    return connex_reg;
}

#else

__global__ void center_shift_kernel(
    cufftComplex* __restrict__ src_batch,
    float* __restrict__ result_out
) {
    int half = FFT_SIZE_REG >> 1;

    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in N/2
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col in N/2

    if (i >= FFT_SIZE_REG || j >= FFT_SIZE_REG) return;

    // Compute index within the padded FFT block
    int shift_i = i ^ half;
    int shift_j = j ^ half;

    // Apply FFT shift (XOR swap quadrants)
    int shifted_idx = shift_i * FFT_SIZE_REG + shift_j;
    float val = src_batch[shifted_idx].x;

    // Compute wrapped output coordinates
    int out_x = (j + GRID_SIZE) % GRID_SIZE;
    int out_y = (i + GRID_SIZE) % GRID_SIZE;    

    result_out[out_y + out_x * GRID_SIZE] = val;
}

static inline void center_shift_cuda(
    cufftComplex* d_resultFFT_reg_batchs,
    float* d_result_out
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (FFT_SIZE_REG  + threadsPerBlock.y - 1) / threadsPerBlock.y);

    center_shift_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_resultFFT_reg_batchs,
        d_result_out
    );

    cudaDeviceSynchronize(); // ou à la fin d'une pipeline
}

__global__ void copy_and_pad_kernel_large(
    cufftComplex *dst,
    const float *image)
{
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (img_x >= FFT_SIZE_REG || img_y >= FFT_SIZE_REG)
        return;

    cufftComplex val = get_value_device(image, img_x, img_y);
    int dst_idx = img_y * FFT_SIZE_REG + img_x;
    dst[dst_idx] = val;
}

static inline void copy_image(
    cufftComplex* dst_dev,           // buffer host de sortie (device)
    const float* image_dev          // image source (device)
    )
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (FFT_SIZE_REG + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copy_and_pad_kernel_large<<<blocksPerGrid, threadsPerBlock>>>(
        dst_dev,
        image_dev
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

int convolve2d_fft_circular(const float* d_image, int* d_roi, float* d_result_out, float* debug){
    //init_FFT_plans(1);
    copy_image(d_imageFFT_reg_batchs,d_image);
    fft2d_batch(d_imageFFT_reg_batchs);
    complex_pointwise_broadcast(d_imageFFT_reg_batchs,d_resultFFT_reg_batchs,1);
    ifft2d_batch(d_resultFFT_reg_batchs);
    center_shift_cuda(d_resultFFT_reg_batchs,d_result_out);
    CHECK_CUDA(cudaGetLastError());
    return 1;
}

#endif
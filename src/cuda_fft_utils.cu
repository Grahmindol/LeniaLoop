#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cuda_fft_utils.h"
#include "config.h"


// 📦 Plans FFT
static cufftHandle fft_plan = 0;
static cufftHandle ifft_plan = 0;
static cudaTextureObject_t tex_kernel = 0;
static int total_elements = FFT_SIZE_REG * FFT_SIZE_REG;

void init_FFT_plans(int batch){
    int n[2] = { FFT_SIZE_REG, FFT_SIZE_REG };
    total_elements = FFT_SIZE_REG * FFT_SIZE_REG * batch;

    if (fft_plan) {
        cufftDestroy(fft_plan);
        fft_plan = 0;
    }

    CHECK_CUFFT(
        cufftPlanMany(&fft_plan,
                    2, n,
                    NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                    NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                    CUFFT_C2C,
                    batch
                )
    );

    if (ifft_plan) {
        cufftDestroy(ifft_plan);
        ifft_plan = 0;
    }
                            
    CHECK_CUFFT(
        cufftPlanMany(&ifft_plan,
                    2, n,
                    NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                    NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                    CUFFT_C2C,
                    batch
                )
    );
    
}

void destroy_FFT_plans() {
    if (fft_plan) { cufftDestroy(fft_plan); fft_plan = 0; }
    if (ifft_plan) { cufftDestroy(ifft_plan); ifft_plan = 0; }
}

void setup_kenel_texture(cufftComplex* d_broadcast_matrix){
     free_kernel_texture();
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_broadcast_matrix;
    resDesc.res.linear.sizeInBytes = FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;  // x et y = 32 bits (float2 = 2 x float)
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;  // on lit directement des float2

    CHECK_CUDA(cudaCreateTextureObject(&tex_kernel, &resDesc, &texDesc, nullptr));
}

void free_kernel_texture(){
    if(tex_kernel)
        CHECK_CUDA(cudaDestroyTextureObject(tex_kernel));
}

void fft2d_batch(cufftComplex* d_data) {
    CHECK_CUFFT(cufftExecC2C(fft_plan, d_data, d_data, CUFFT_FORWARD));
}

__global__ void normalize_ifft(cufftComplex* data, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        data[idx].x /= FFT_SIZE_REG*FFT_SIZE_REG;
        data[idx].y /= FFT_SIZE_REG*FFT_SIZE_REG;
    }
}

void ifft2d_batch(cufftComplex* d_data) {
    
    CHECK_CUFFT(cufftExecC2C(ifft_plan, d_data, d_data, CUFFT_INVERSE));

    dim3 block(1024);
    dim3 grid((total_elements + block.x - 1) / block.x);
    normalize_ifft<<<grid, block>>>(d_data, total_elements);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernel_pointwise_batch_broadcast(
    cufftComplex* __restrict__ A,
    cufftComplex* __restrict__ C,
    cudaTextureObject_t tex_B,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = FFT_SIZE_REG * FFT_SIZE_REG;
    int total_size = matrix_size * batch;

    if (idx < total_size) {
        int local_idx = idx % matrix_size; // position dans une matrice n×n
        cufftComplex a = A[idx];
        cufftComplex b = tex1Dfetch<cufftComplex>(tex_B, local_idx); // lecture via texture


        C[idx].x = a.x * b.x - a.y * b.y;
        C[idx].y = a.x * b.y + a.y * b.x;
    }
}

/**
 * @brief Performs pointwise complex multiplication with broadcasting.
 * 
 * This function computes the element-wise multiplication of two complex matrices, 
 * where the second matrix `d_broadcast_matrix` is broadcasted across the batch dimension of `d_input_batch`.
 * 
 * @param d_input_batch  Pointer to the input complex matrix batch [batch x n x n].
 * @param d_broadcast_matrix  Pointer to the 2D complex matrix [n x n] that is broadcasted to all batches.
 * @param d_output_batch  Pointer to the output complex matrix batch [batch x n x n].
 * @param n  The dimension of each square matrix (n x n).
 * @param batch_size  The number of matrices in the batch.
 * 
 * @note 1.246 ms during the last test.
 */
void complex_pointwise_broadcast(
    cufftComplex* d_input_batch,      // [batch_size x FFT_SIZE_REG x FFT_SIZE_REG] Input batch of complex matrices
    cufftComplex* d_output_batch,     // [batch_size x FFT_SIZE_REG x FFT_SIZE_REG] Output batch of complex matrices
    int batch_size                    // The number of matrices in the batch
) {
    int total_size = FFT_SIZE_REG * FFT_SIZE_REG * batch_size;  // Total number of elements across the entire batch
    int block_size = 1024;                 // Number of threads per block
    int num_blocks = (total_size + block_size - 1) / block_size;  // Calculate number of blocks

    // Launch the kernel to compute the element-wise multiplication
    kernel_pointwise_batch_broadcast<<<num_blocks, block_size>>>(
        d_input_batch,
        d_output_batch,
        tex_kernel,
        batch_size
    );

    // Check for CUDA errors and synchronize device
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}



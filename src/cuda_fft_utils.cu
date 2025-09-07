#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cuda_fft_utils.h"
#include "config.h"

// ======================================================
// 🔹 Plans FFT et texture kernel
// ======================================================
static cufftHandle fft_plan = 0;
static cufftHandle ifft_plan = 0;
static cudaTextureObject_t tex_kernel = 0;
static int total_elements = FFT_SIZE_REG * FFT_SIZE_REG;

// ======================================================
// 📦 Initialisation et destruction des plans FFT
// ======================================================

/**
 * @brief Initialise les plans FFT et IFFT pour un batch donné.
 * 
 * @param batch Nombre de matrices à traiter en batch.
 */
void init_FFT_plans(int batch) {
    int n[2] = { FFT_SIZE_REG, FFT_SIZE_REG };
    total_elements = FFT_SIZE_REG * FFT_SIZE_REG * batch;

    // Destruction des plans existants
    if (fft_plan) { cufftDestroy(fft_plan); fft_plan = 0; }
    if (ifft_plan) { cufftDestroy(ifft_plan); ifft_plan = 0; }

    // Création du plan FFT
    CHECK_CUFFT(cufftPlanMany(&fft_plan,
                               2, n,
                               NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                               NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                               CUFFT_C2C,
                               batch));

    // Création du plan IFFT
    CHECK_CUFFT(cufftPlanMany(&ifft_plan,
                               2, n,
                               NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                               NULL, 1, FFT_SIZE_REG * FFT_SIZE_REG,
                               CUFFT_C2C,
                               batch));
}

/**
 * @brief Libère les plans FFT et IFFT.
 */
void destroy_FFT_plans() {
    if (fft_plan) { cufftDestroy(fft_plan); fft_plan = 0; }
    if (ifft_plan) { cufftDestroy(ifft_plan); ifft_plan = 0; }
}

// ======================================================
// 📦 Gestion de la texture CUDA pour le kernel
// ======================================================

/**
 * @brief Crée la texture CUDA à partir d'une matrice kernel en device memory.
 */
void setup_kernel_texture(cufftComplex* d_broadcast_matrix) {
    free_kernel_texture();

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_broadcast_matrix;
    resDesc.res.linear.sizeInBytes = FFT_SIZE_REG * FFT_SIZE_REG * sizeof(cufftComplex);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    CHECK_CUDA(cudaCreateTextureObject(&tex_kernel, &resDesc, &texDesc, nullptr));
}

/**
 * @brief Libère la texture kernel.
 */
void free_kernel_texture() {
    if (tex_kernel) {
        CHECK_CUDA(cudaDestroyTextureObject(tex_kernel));
        tex_kernel = 0;
    }
}

// ======================================================
// 📦 FFT / IFFT
// ======================================================

/**
 * @brief Exécute la FFT 2D sur un batch de matrices.
 */
void fft2d_batch(cufftComplex* d_data) {
    CHECK_CUFFT(cufftExecC2C(fft_plan, d_data, d_data, CUFFT_FORWARD));
}

/**
 * @brief Normalisation post-IFFT (kernel CUDA).
 */
__global__ void normalize_ifft(cufftComplex* data, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        data[idx].x /= FFT_SIZE_REG * FFT_SIZE_REG;
        data[idx].y /= FFT_SIZE_REG * FFT_SIZE_REG;
    }
}

/**
 * @brief Exécute l'IFFT 2D sur un batch et normalise le résultat.
 */
void ifft2d_batch(cufftComplex* d_data) {
    CHECK_CUFFT(cufftExecC2C(ifft_plan, d_data, d_data, CUFFT_INVERSE));

    dim3 block(1024);
    dim3 grid((total_elements + block.x - 1) / block.x);
    normalize_ifft<<<grid, block>>>(d_data, total_elements);
    CHECK_CUDA(cudaGetLastError());
}

// ======================================================
// 📦 Multiplication complexe pointwise avec broadcasting
// ======================================================

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
        int local_idx = idx % matrix_size;
        cufftComplex a = A[idx];
        cufftComplex b = tex1Dfetch<cufftComplex>(tex_B, local_idx);

        // Multiplication complexe
        C[idx].x = a.x * b.x - a.y * b.y;
        C[idx].y = a.x * b.y + a.y * b.x;
    }
}

/**
 * @brief Effectue la multiplication complexe pointwise avec broadcast sur un batch de matrices.
 *
 * @param d_input_batch      Batch de matrices complexes d'entrée [batch x FFT_SIZE_REG x FFT_SIZE_REG].
 * @param d_output_batch     Batch de matrices complexes de sortie [batch x FFT_SIZE_REG x FFT_SIZE_REG].
 * @param batch_size         Nombre de matrices dans le batch.
 */
void complex_pointwise_broadcast(
    cufftComplex* d_input_batch,
    cufftComplex* d_output_batch,
    int batch_size
) {
    int total_size = FFT_SIZE_REG * FFT_SIZE_REG * batch_size;
    int block_size = 1024;
    int num_blocks = (total_size + block_size - 1) / block_size;

    kernel_pointwise_batch_broadcast<<<num_blocks, block_size>>>(
        d_input_batch,
        d_output_batch,
        tex_kernel,
        batch_size
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

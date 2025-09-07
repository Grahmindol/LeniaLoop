#ifndef CUDA_FFT_UTILS_H
#define CUDA_FFT_UTILS_H

#include <cufft.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes forward and inverse FFT plans for batched 2D FFT.
 *
 * @param batch Number of matrices to process in the batch.
 */
void init_FFT_plans(int batch);

/**
 * @brief Creates a CUDA texture object for a broadcast kernel matrix.
 *
 * This texture is used for efficient read-only access in pointwise
 * multiplication on the GPU.
 *
 * @param d_broadcast_matrix Pointer to device memory containing the complex kernel matrix.
 */
void setup_kernel_texture(cufftComplex* d_broadcast_matrix);

/**
 * @brief Destroys both FFT plans (forward and inverse) if they exist.
 */
void destroy_FFT_plans(void);

/**
 * @brief Frees the CUDA texture object for the kernel if it exists.
 */
void free_kernel_texture(void);

/**
 * @brief Performs a batched 2D forward FFT on a set of complex matrices.
 *
 * @param d_data Pointer to device memory containing the input batch.
 */
void fft2d_batch(cufftComplex* d_data);

/**
 * @brief Performs a batched 2D inverse FFT on a set of complex matrices.
 *
 * After computing the inverse FFT, the results are normalized.
 *
 * @param d_data Pointer to device memory containing the input batch.
 */
void ifft2d_batch(cufftComplex* d_data);

/**
 * @brief Performs element-wise complex multiplication with broadcasting.
 *
 * Multiplies each matrix in the input batch with the broadcasted kernel
 * matrix stored in the texture object, writing results to the output batch.
 *
 * @param im Input batch of complex matrices on device.
 * @param re Output batch of complex matrices on device.
 * @param batch_count Number of matrices in the batch.
 */
void complex_pointwise_broadcast(cufftComplex* im, cufftComplex* re, int batch_count);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FFT_UTILS_H

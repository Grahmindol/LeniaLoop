#ifndef CUDA_CONVOLVE_UTILS_H
#define CUDA_CONVOLVE_UTILS_H

#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846

/// Macros utilitaires
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes buffers needed for FFT-based convolution on the GPU.
 */
void init_convolve_buffer(void);

/**
 * @brief Precomputes the FFT of a convolution kernel.
 *
 * The kernel is copied to GPU memory and its FFT is stored for later use
 * in pointwise multiplication.
 *
 * @param kernel Pointer to the host kernel array (size x size).
 * @param size The width/height of the square kernel.
 */
void precompute_kernelFFT(const float *kernel, int size);

/**
 * @brief Frees all GPU buffers allocated for convolution.
 */
void destroy_convolve_buffer(void);

/**
 * @brief Performs 2D circular convolution of an image (using FFT. if enabled)
 *
 * @param d_image Pointer to the input image on device (flattened float array).
 * @param h_sub_region_interst Pointer to host array defining sub-region of interest.
 * @param d_result_out Pointer to output buffer on device.
 * @param debug Optional pointer for storing debug info (can be NULL).
 * @return An integer status code (0 if success, non-zero on error).
 */
int convolve2d_circular(const float* d_image, int* h_sub_region_interst, float* d_result_out, float* debug);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CONVOLVE_UTILS_H

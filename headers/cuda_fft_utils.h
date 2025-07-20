#ifndef CUDA_FFT_UTILS_H
#define CUDA_FFT_UTILS_H

#include <cufft.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
void init_FFT_plans(int batch);
void setup_kenel_texture(cufftComplex* d_broadcast_matrix);
void destroy_FFT_plans(void);
void free_kernel_texture();

void fft2d_batch(cufftComplex* d_data);
void ifft2d_batch(cufftComplex* d_data);
void complex_pointwise_broadcast(cufftComplex* im, cufftComplex* re, int batch_count);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FFT_UTILS_H
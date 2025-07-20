#ifndef CUDA_CONVOLVE_UTILS_H
#define CUDA_CONVOLVE_UTILS_H

#include <cufft.h>
#include <cuda_runtime.h>

#include <math.h>
#define PI 3.14159265358979323846

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif
void init_convolve_buffer();
void precompute_kernelFFT(const float *kernel, int size);
void destroy_convolve_buffer();

int convolve2d_fft_circular(const float* d_image, int* h_sub_region_interst, float* d_result_out, float* debug);
#ifdef __cplusplus
}
#endif

#endif // CUDA_CONVOLVE_UTILS_H
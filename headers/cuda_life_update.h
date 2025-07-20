#ifndef LIFE_UPDATE_H
#define LIFE_UPDATE_H


#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_life_gpu(int current,float* h_pixels_result);
void destroy_cuda_update();
void init_cuda_update(float* h_pixels );

#ifdef __cplusplus
}
#endif

#endif // LIFE_UPDATE_H

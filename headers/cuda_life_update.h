#ifndef LIFE_UPDATE_H
#define LIFE_UPDATE_H

#include <cuda_runtime.h>

/**
 * @file life_update.h
 * @brief GPU-based update system for Life-like cellular automata.
 *
 * This module manages GPU memory and kernels to simulate
 * the evolution of a grid using convolution-based neighbor counting
 * and smooth growth rules.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize CUDA resources for the life update system.
 *
 * This function allocates GPU memory for:
 *  - Multiple frame buffers (double or triple buffering depending on BUFFER_NUMBER).
 *  - Convolution buffer for neighbor counts.
 *  - Region-of-interest (ROI) buffer, if REGION_SIZE is defined.
 *
 * @param h_pixels Pointer to the initial pixel data on the host.
 *                 Size must be `GRID_SIZE * GRID_SIZE * 3` floats.
 */
void init_cuda_update(float* h_pixels);

/**
 * @brief Update the grid state on the GPU.
 *
 * Performs the following steps:
 *  - Run 2D FFT-based convolution to compute neighbor counts.
 *  - Apply the update kernel to compute the next state.
 *  - Optionally copy back results to the host after a given number of iterations.
 *  - Update subregion interest maps (ROI) if enabled.
 *
 * @param current Index of the current frame buffer (0..BUFFER_NUMBER-1).
 * @param h_pixels_result Pointer to a host buffer where results will be copied.
 *                        Must have size `GRID_SIZE * GRID_SIZE * 3` floats.
 */
void update_life_gpu(int current, float* h_pixels_result);

/**
 * @brief Free all allocated CUDA resources.
 *
 * This function must be called once you are done using the GPU life update system.
 * It will release all GPU memory allocated in `init_cuda_update()`.
 */
void destroy_cuda_update();

#ifdef __cplusplus
}
#endif

#endif // LIFE_UPDATE_H

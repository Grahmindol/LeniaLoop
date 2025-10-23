#ifndef LIFE_UPDATE_H
#define LIFE_UPDATE_H

#include <cuda_runtime.h>

/**
 * @file life_updater.h
 * @brief update system for Life-like cellular automata.
 *
 * This module manages the evolution of a grid using convolution-based neighbor counting
 * and smooth rules.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize resources for the life update system.
 *
 * @param h_pixels Pointer to the initial pixel data on the host.
 *                 Size must be `GRID_SIZE * GRID_SIZE * 3` floats.
 */
void init_updater(float* h_pixels);

/**
 * @brief Update the grid state.
 *
 * @param current Index of the current frame buffer (0..BUFFER_NUMBER-1).
 * @param h_pixels_result Pointer to a host buffer where results will be copied.
 *                        Must have size `GRID_SIZE * GRID_SIZE * 3` floats.
 */
void update_life(int current, float* h_pixels_result);

/**
 * @brief Free all allocated resources.
 *
 * This function must be called once you are done using the life update system.
 * It will release all memory allocated in `init_cuda_update()`.
 */
void destroy_updater();

#ifdef __cplusplus
}
#endif

#endif // LIFE_UPDATE_H

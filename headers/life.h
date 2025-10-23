#ifndef LIFE_H
#define LIFE_H

#include <GL/glut.h>

#include "cuda_convolve_utils.h"
#include "config.h"

/**
 * @brief Global pixel buffer storing the current frame.
 *
 * The buffer stores RGB float values for each cell in the simulation grid.
 * Size: GRID_SIZE x GRID_SIZE x 3 (RGB channels).
 */
extern float* pixels;

/**
 * @brief Initializes the Game of Life simulation.
 *
 * Allocates memory for the pixel buffer, loads an initial pattern (from a PNG
 * file, a random seed, or the built-in pattern), initialize updater
 */
void init_life();

/**
 * @brief Frees resources used by the Game of Life simulation.
 *
 * Frees the pixel buffer and destroys any resources used for updating
 * the simulation.
 */
void destroy_life();

/**
 * @brief Updates the simulation by one frame.
 *
 * Performs a full update of the grid state using the uppdater.
 *
 * @param frame Current frame number (can be used for temporal effects).
 */
void life_update_frame(int frame);

#endif // LIFE_H

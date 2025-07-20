#ifndef LIFE_H
#define LIFE_H

#include <GL/glut.h>

#include "cuda_convolve_utils.h"
#include "config.h"


// Declare the external pixel buffers (2 frames in RGB format)
extern float* pixels;

// Function prototypes
void init_life();
void destroy_life();
void life_update_frame(int frame);

#endif // LIFE_H

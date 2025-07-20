#include "../headers/life.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <time.h>
#include <cuda_runtime.h>
#include "cuda_life_update.h"

#include "png_util.h"
#include "csv_util.h"


// Global variables for pixel buffers and state tracking
float* pixels = NULL;
float kernel[2*RADIUS +1 ][2*RADIUS + 1];

// array used for debug displayes as green chanel
float debug_grid[GRID_SIZE][GRID_SIZE];

static inline float bell(float x, float m, float s) {
    return expf(-powf((x - m) / s, 2) / 2);
}

static inline float growth(float x){
    if (x > 0.3f) return -1;
    return 2*(bell(x,0.15f,0.015f))-1;
}

static inline float K_cross(int dx, int dy){
    float r = powf((float)(dx*dx + dy*dy),0.5f)/RADIUS;
    return bell(r,0.5,0.15);
}


void init_kernel(){
    double sum = 0;
    for (int dx = -RADIUS; dx < RADIUS+1; dx++)
    for (int dy = -RADIUS; dy < RADIUS+1; dy++)
        sum += kernel[dx+RADIUS][dy+RADIUS] = K_cross(dx,dy);
    for (int dx = -RADIUS; dx < RADIUS+1; dx++)
    for (int dy = -RADIUS; dy < RADIUS+1; dy++)
        kernel[dx+RADIUS][dy+RADIUS] /= sum;

    init_convolve_buffer();
    precompute_kernelFFT((float*)kernel,2*RADIUS+1);
}


void init_life()
{
     // Clear both frame buffers
    pixels = malloc(GRID_SIZE * GRID_SIZE * 3 * sizeof(float));
 
    memset(pixels, 0, GRID_SIZE * GRID_SIZE * 3 * sizeof(float));

    if (!pixels) {
        fprintf(stderr, "Failed to allocate memory for pixel buffers.\n");
        exit(EXIT_FAILURE);
    }

    #ifdef IMG_PATH
    
    int imgWidth, imgHeight;
    unsigned char *imgData = read_png_file(IMG_PATH, &imgWidth, &imgHeight);
    if (!imgData) {
        perror("Failed to load PNG");
        exit(EXIT_FAILURE);
    }

    // Calculate starting position to center the image
    int startX = (GRID_SIZE - imgWidth) / 2;
    int startY = (GRID_SIZE - imgHeight) / 2;

    // Copy the image pixels into the grid (only the red channel is set)
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            int imgIndex = (y * imgWidth + x) * 3;
            int screenIndex = ((startY + y) * GRID_SIZE + (startX + x)) * 3;

            if (screenIndex < GRID_SIZE * GRID_SIZE * 3 && imgIndex < imgWidth * imgHeight * 3) {
                pixels[screenIndex]     = ((float) imgData[imgIndex]) / 255.0f;  // Red channel
                //pixels[screenIndex + 1] = 0.0f;                  // Green channel
                //pixels[screenIndex + 2] = 0.0f;                  // Blue channel
            }
        }
    }

    free(imgData);

    #elif RANDOM_SEED
    srand(RANDOM_SEED);
    // Copy the image pixels into the grid (only the red channel is set)
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            int screenIndex = (y * GRID_SIZE + x) * 3;
            pixels[screenIndex]     = (float)rand() / RAND_MAX;  // Red channel
            pixels[screenIndex + 1] = 0.0f;                  // Green channel
            pixels[screenIndex + 2] = 0.0f;                  // Blue channel
        }
    }
    #else

    int scale = RADIUS / 13;
    int imgWidth = 20 * scale, imgHeight = 20 * scale;
    float imgData[20][20] = {
        {0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0},
        {0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0},
        {0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0},
        {0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0},
        {0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0.18,0.42,0,0},
        {0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0},
        {0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0},
        {0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0},
        {0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0},
        {0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07},
        {0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11},
        {0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1},
        {0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05},
        {0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01},
        {0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0},
        {0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0},
        {0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0},
        {0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0},
        {0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0},
        {0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0}
    };
   

    // Calculate starting position to center the image
    int startX = (GRID_SIZE - imgWidth) / 2;
    int startY = (GRID_SIZE - imgHeight) / 2;

    // Copy the image pixels into the grid (only the red channel is set)
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            int screenIndex = ((startY + y) * GRID_SIZE + (startX + x)) * 3;

            if (screenIndex < GRID_SIZE * GRID_SIZE * 3) {
                pixels[screenIndex]       = imgData[x/scale][y/scale];  // Red channel
                //pixels[screenIndex + 1] = 0.0f;                  // Green channel
                //pixels[screenIndex + 2] = 0.0f;                  // Blue channel
            }
        }
    }

    #endif

    init_kernel();
    init_cuda_update(pixels);
    
}

void destroy_life(){
    free(pixels);
    destroy_cuda_update();
}

// Update the entire grid state.
void life_update_frame(int currentFrame)
{
    update_life_gpu(currentFrame,pixels);
    //save_to_csv("data_pixel_center",-1, pixels[GRID_SIZE * GRID_SIZE / 2 + GRID_SIZE/2]);
}

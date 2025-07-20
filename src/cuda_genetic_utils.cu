#include "cuda_genetic_utils.h"
#include "config.h"

#include <cuda.h>
#include <curand_kernel.h> 
#include <stdio.h>        
#include <stdlib.h>         
#include <time.h>          

#define GENETIC_BATCH 10
float best_score = -INFINITY;
float* d_best;

void init_genetic(float* d_pixels){
    cudaMalloc((void**)&d_best,sizeof(float*)* GRID_SIZE * GRID_SIZE * 3);
    cudaMemcpy(d_best, d_pixels, sizeof(float) * GRID_SIZE * GRID_SIZE * 3, cudaMemcpyDeviceToDevice);
}

void destroy_genetic(){
    cudaFree(d_best);
}

__global__ void mutate_kernel(float* grid, float* d_best, float mutation_strength, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = GRID_SIZE * GRID_SIZE;
    if (idx >= total) return;

    // Init RNG
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Applica una mutazione locale
    float mutation = (curand_uniform(&state) - 0.5f) * 2.0f * mutation_strength;  // valore in [-strength, +strength]
    float new_val = d_best[idx*3] + mutation;

    // Clamp tra 0 e 1
    grid[idx*3] = fminf(1.0f, fmaxf(0.0f, new_val));
}


void mutate(float* d_data){
    int total = GRID_SIZE * GRID_SIZE;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mutate_kernel<<<blocks, threads>>>(d_data,d_best, 0.3f, time(NULL));
    //printf("boom\n");
}

float evaluate(float* d_data, int connex_reg){
    return 0;//d_data[GRID_SIZE*GRID_SIZE / 2];
}

void change_candidate(float* d_c , int connex_reg){
    static int iteration = 0;
    if(iteration>10) iteration = 0;
    iteration++;

    printf("%f \n",evaluate(d_c, connex_reg));
    mutate(d_c);
}
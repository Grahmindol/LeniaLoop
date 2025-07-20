#ifndef CUDA_GENETIC_UTILS_H
#define CUDA_GENETIC_UTILS_H

void change_candidate(float* c , int connex_reg);
void init_genetic(float* d);
void destroy_genetic();

#endif
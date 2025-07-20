#include "config.h"

#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#ifdef RECORD_CSV


#define MAX_ITERATION 1000

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#else
#include <stdio.h>
#include <stdlib.h>
#endif

void save_to_csv(const char *filename, int iteration,float data) {
    static int iter = 0;
    if(iteration >= 0){
        iter = iteration;
    }

    // Créer le chemin complet du fichier
    char path[256];
    snprintf(path, sizeof(path), "csv/%s.csv", filename);

    FILE *fp = fopen(path, iteration == 0 ? "w" : "a"); // Write for first iter, append after
    if (!fp) {
        perror("Error opening file");
        return;
    }

    if (iteration == 0) {
        fprintf(fp, "0"); // First iteration starts row
    } else {
        fprintf(fp, "\n%d", iteration);
    }

    fprintf(fp, ";%.6f", data); // Append new column

    iter++;
    fclose(fp);
}

#define RECORD_CPU_TIMING_BEGIN(label) clock_t label##_start = clock();
#define RECORD_CPU_TIMING_END(label) clock_t label##_end = clock(); \
    save_to_csv(#label,-1,1000.0 * (label##_end - label##_start) / CLOCKS_PER_SEC);



#else
#define RECORD_CPU_TIMING_BEGIN(label)
#define RECORD_CPU_TIMING_END(label)
#endif

#endif
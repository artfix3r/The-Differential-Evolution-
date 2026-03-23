#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

// ----------------------------------------------------------------------------------------------------------------------------# @blk1
// Objective function
// 0: Levy 3-dimensional
// 1: Shifted Rastigrin's  Function
// 2: Shifted Rosenbrock's Function
// 3: Shifted Griewank's   Function
// 4: Shifted Sphere's     Function
const int   SELECTED_OBJ_FUNC = 0;
const float START_RANGE_MIN   = -5.12f;
const float START_RANGE_MAX   =  5.12f;
const float OMEGA             = 0.5;
const float c1                = 1.5;
const float c2                = 1.5;
const float phi               = 3.1415;

#define arridim(ARR)  (sizeof((ARR)) / sizeof((*(ARR))))

#define gpuErrorCheck(ST){  gpuAssert((ST), __FILE__, __LINE__);  }
void gpuAssert(cudaError_t code, const char *file, int line);

float getRandom(float low, float high);
float getRandomClamped();
extern "C" __device__ __host__ float fitness_function(int, float* x);
extern "C"                     void  cuda_pso(int,int,int, float*,float*,float*,float*);  // host entry point for the gpu

/* Copyright 2017 Ian Rankin */
// This class is a wrapper to make calls to the cuda differential evolution code easier to work with.
// It handles all of the internal memory allocation for the differential evolution and holds them as device memory for the GPU
//
// Example wrapper usage:
//
// float mins[3] = {0,-1,-3.14};
// float maxs[3] = {10,1,3.13};
// DifferentialEvolution minimizer(64,100, 3, 0.9, 0.5, mins, maxs);
// minimizer.fmin(NULL);
//
//////////////////////////////////////////////////////////////////////////////////////////////
// However if needed to pass arguements then an example usage is:
//
// // create the min and max bounds for the search space.
// float minBounds[2] = {-50, -50};
// float maxBounds[2] = {100, 200};
//
// // a random array or data that gets passed to the cost function.
// float arr[3] = {2.5, 2.6, 2.7};
//
// // data that is created in host, then copied to a device version for use with the cost function.
// struct data x;
// struct data *d_x;
// gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
// unsigned long size = sizeof(struct data);
// gpuErrorCheck(cudaMalloc((void **)&d_x, size));
// x.v   = 3;
// x.dim = 2;
// gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
//
// // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 0.5
// DifferentialEvolution minimizer(192,50, 2, 0.9, 0.5, minBounds, maxBounds);
// gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
//
// // get the result from the minimizer
// std::vector<float> result = minimizer.fmin(d_x);
#ifndef DifferentialEvolution_hpp
#define DifferentialEvolution_hpp

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "DifferentialEvolutionGPU.h"

#define arridim(ARR)  (sizeof((ARR)) / sizeof((*(ARR))))

struct data{  // a helper struct to send data to the cost
	float* arr;  //    arbitrary arguments for one (and only one) of the cost functions?
	float  v;    // an arbitrary argument  for one (and only one) of the cost functions?
	int    dim;  // dimensionality of the problem
};

class DifferentialEvolution{
private:
	float* d_target1;
	float* d_target2;
	float* d_cost;
	float* d_mutant;
	float* d_trial;
	float* d_min;
	float* d_max;
	void*  d_randStates;
	float* h_cost;

	int   popSize;
	int   numGenerations;
	int   dim;
	int   CR;
	float F;

public:
	// Constructor for DifferentialEvolution
	// @param PopulationSize:    the number of agents the DE solver uses.
	// @param NumGenerations:    the number of generation the differential evolution solver uses.
	// @param Dimensions:        the number of dimesnions for the solution.
	// @param crossoverConstant: the number of mutants allowed pass each generation (CR in literature) given in the range [0,1]
	// @param mutantConstant:    the scale on mutant changes (F in literature) given [0,2] default = 0.5
	// @param func:              the cost function to minimize.
	DifferentialEvolution(int PopulationSize, int NumGenerations, int Dimensions, float crossoverConstant, float mutantConstant, float* minBounds, float* maxBounds);
	~DifferentialEvolution();

	// fmin. wrapper to the cuda function C function for differential evolution.
	// @param args: this a pointer to arguments for the cost function. This MUST point to device memory or NULL.
	// @return the best set of parameters
	std::vector<float> fmin(void* args);
};

#endif /* DifferentialEvolution_hpp */

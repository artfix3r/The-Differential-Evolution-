/* Copyright 2017 Ian Rankin
This is a test code to show an example usage of Differential Evolution
*/
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "DifferentialEvolution.hpp"

extern float quadraticFunc(const float* vec, const void* args);

// @param nruns: number of runs, in order to calculate the mean and standard deviation after @nruns
// @param D:     dimensionality of the problem
// @param P:     population size
// @param G:     number of generations
void run(int nruns, int DN,int PN,int Di,int Pi, int D,int P,int G, float CR,float F){
	printf("  {\n");
	printf("    \"dim\": %d,\n",  D);  // printf("\x1b[92m%s  \x1b[0mpop \x1b[31m%'5d  \x1b[0mdim \x1b[32m%'3d  \x1b[0mgen \x1b[34m%'5d\x1b[0m\n", __func__,P,D,G);
	printf("    \"pop\": %d,\n",  P);
	printf("    \"gen\": %d,\n",  G);
	printf("    \"CR\": %.3f,\n", CR);
	printf("    \"F\": %.3f,\n",  F);
	printf("    \"rnuns\": %d,\n", nruns);

	float  minBound  = -5.12;  // single/global/constant min for all dimensions in the parameter/search space
	float  maxBound  = +5.12;  // single/global/constant max for all dimensions in the parameter/search space
	float* minBounds = (float*)malloc(sizeof(float)*D);
	float* maxBounds = (float*)malloc(sizeof(float)*D);
	for(int i=0; i<D; ++i){
		minBounds[i] = minBound;
		maxBounds[i] = maxBound;
	}

	// data that is created in host, then copied to a device version for use with the cost function. these data/x/args are only used in @costWithArgs()?
	float arr[3] = {2.5,2.6,2.7};  // a random array of data that gets passed to the cost function
	struct data  args;
	struct data* d_args;
	args.v   = 3;  // an arbitrary value...
	args.dim = D;
	gpuErrorCheck(cudaMalloc(        &args.arr,     sizeof(arr)));
	gpuErrorCheck(cudaMalloc((void**)&d_args,       sizeof(struct data)));
	gpuErrorCheck(cudaMemcpy(args.arr,(void*)&arr,  sizeof(arr),         cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_args,  (void*)&args, sizeof(struct data), cudaMemcpyHostToDevice));

	float* z_mean = (float*)malloc(sizeof(float)*D);
	float* z_std  = (float*)malloc(sizeof(float)*D);
	float* sum    = (float*)calloc(D,sizeof(float));  // running sum            for each coordinate
	float* sumsq  = (float*)calloc(D,sizeof(float));  // running sum of squares for each coordinate
	float  y_mean;
	float  y_std;
	float  y_sum      = 0;  // running sum            for the minimum function value
	float  y_sumsq    = 0;  // running sum of squares for the minimum function value

	for(int run=0; run<nruns; ++run){
		DifferentialEvolution minimizer(P,G,D, CR,F, minBounds,maxBounds);  // create the minimizer with a popsize of P, G generations, D dimensions
		std::vector<float> x = minimizer.fmin(d_args);  // get the @x from the minimizer, where @x is the vector that maximizes the cost function?
		for(int i=0; i<D; ++i){
			float xi  = x[i];
			sum[i]   += xi;
			sumsq[i] += xi*xi;
			z_mean[i] = sum[i]  /(run+1);
			z_std[i]  = sumsq[i]/(run+1) - z_mean[i]*z_mean[i];  // printf("result ");  for(int i=0; i<D; ++i) printf(" %6.3f", xi);  putchar(0x0a);
		}
		float y  = quadraticFunc(x.data(),(void*)&args);
		// y_sum   += y;
		// y_sumsq += y*y;
		// y_mean   = y_sum  /(run+1);
		// y_std    = y_sumsq/(run+1) - y_mean*y_mean;  // printf("gBest ");  for(int k=0; k<D; ++k) printf(" %6.3f", gBest);  putchar(0x0a);
	}
	printf("    \"x_mean\":[");  for(int i=0; i<D-1; ++i) printf("%6.3f,", z_mean[i]);  printf("%6.3f", z_mean[D-1]); printf("],\n");
	printf("    \"x_std\": [");  for(int i=0; i<D-1; ++i) printf("%6.3f,", z_std[i]);   printf("%6.3f", z_std[D-1]);  printf("],\n");
	printf("    \"y_mean\": %6.3f,\n", y_mean);
	printf("    \"y_std\":  %6.3f\n",  y_std);
	printf("  }");
	if(!(Di==DN-1 && Pi==PN-1))  printf(",");
	putchar(0x0a);

	cudaFree(args.arr);
	cudaFree(d_args);
	free(minBounds);
	free(maxBounds);
	free(z_mean);
	free(z_std);
	free(sum);
	free(sumsq);
}

int main(int nargs, char* args[]){
	int   NRUNS = 25;  // number of runs for each configuration, in order to calculate the mean and standard deviation
	// int   D[]   = {2,3};
	// int   P[]   = {50,100,1024};
	// int   D[]   = {2, 5,10};  // dimensionality of the problem
	// int   P[]   = {5,10,50};  // population size
	int   D[]   = {10, 50,100};  // dimensionality of the problem
	int   P[]   = {50,100,500};  // population size
	float CR    = 0.9;
	float F     = 0.8;

	puts("[");
	for(int i=0; i<arridim(D); ++i)
		for(int j=0; j<arridim(P); ++j)
			run(NRUNS, arridim(D),arridim(P),i,j, D[i],P[j],104*D[i], CR,F);
	puts("]");
	return 0;
}

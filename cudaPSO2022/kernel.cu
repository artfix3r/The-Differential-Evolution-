#include "kernel.h"

void gpuAssert(cudaError_t code, const char *file, int line){  // basic function for exiting code on CUDA errors.
	if(code!=cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

float getRandom(float low, float high){  return low + float(((high - low) + 1)*rand()/(RAND_MAX + 1.0));  }  // Obtenir un random entre low et high
float getRandomClamped(){                return (float) rand()/(float) RAND_MAX;  }                          // Obtenir un random entre 0.0f and 1.0f inclusif

// ----------------------------------------------------------------------------------------------------------------------------# @blk1

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
__device__ __host__ float fitness_function(int num_dimensions, float* x){
	float res     = 0;
	float somme   = 0;
	float produit = 0;

	switch(SELECTED_OBJ_FUNC){
		case 0:{
			float y1 = 1 + (x[0] - 1)/4;
			float yn = 1 + (x[num_dimensions-1] - 1)/4;
			res += pow(sin(phi*y1), 2);
			for(int i = 0; i < num_dimensions-1; i++) {
				float y  = 1 + (x[i] - 1)/4;
				float yp = 1 + (x[i+1] - 1)/4;
				res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
			}
		}break;
		case 1:{
			for(int i = 0; i < num_dimensions; i++) {
				float zi = x[i] - 0;
				res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
			}
			res -= 330;
		}break;
		case 2:{
			for(int i = 0; i < num_dimensions-1; i++) {
				float zi = x[i] - 0 + 1;
				float zip1 = x[i+1] - 0 + 1;
				res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
			}
			res += 390;
		}break;
		case 3:{
			for(int i = 0; i < num_dimensions; i++) {
				float zi = x[i] - 0;
				somme += pow(zi, 2)/4000;
				produit *= cos(zi/pow(i+1, 0.5));
			}
			res = somme - produit + 1 - 180;
		}break;
		case 4:{
			for(int i = 0; i < num_dimensions; i++) {
				float zi = x[i] - 0;
				res += pow(zi, 2);
			}
			res -= 450;
		}break;
	}

	return res;
}

// Runs on the GPU, called from the CPU or the GPU
__global__ void kernelUpdateParticle(int num_dimensions,int num_particles, float* positions,float* velocities, float* pBests,float* gBest, float r1,float r2){
	int i = blockIdx.x * blockDim.x + threadIdx.x;  if(i >= num_particles * num_dimensions)  return;  // avoid an out of bound for the array

	// float rp = getRandomClamped();
	// float rg = getRandomClamped();
	float rp = r1;  // random weight for personnal =>  computed from @getRandomClamped
	float rg = r2;  // random weight for global =>  computed from @getRandomClamped

	velocities[i] = OMEGA * velocities[i] +  c1 * rp * (pBests[i] - positions[i]) +  c2 * rg * (gBest[i % num_dimensions] - positions[i]);  // Mise à jour de velocities et positions
	positions[i] += velocities[i];  // Update posisi particle. Mise à jour de la position de la particule courante incrémentant la position de la particule courante avec la vitesse de la particule courante
}

// Runs on the GPU, called from the CPU or the GPU
__global__ void kernelUpdatePBest(int num_dimensions,int num_particles, float* positions,float* pBests,float* gBest, float* tempParticle1,float* tempParticle2){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= num_particles * num_dimensions || i % num_dimensions != 0)  return;

	for(int j = 0; j < num_dimensions; j++){
		tempParticle1[j] = positions[i + j];
		tempParticle2[j] = pBests[i + j];
	}

	if(fitness_function(num_dimensions, tempParticle1) < fitness_function(num_dimensions, tempParticle2)){
		for(int k = 0; k < num_dimensions; k++)
			pBests[i + k] = positions[i + k];
	}
}

extern "C" void cuda_pso(int num_dimensions,int num_particles,int max_iter, float* positions,float* velocities,float* pBests,float* gBest){  // host entry point for the gpu
  // declare all the arrays on the device
	int size = num_particles * num_dimensions;
	float* devPos;         cudaMalloc((void**)&devPos,        sizeof(float)*size);
	float* devVel;         cudaMalloc((void**)&devVel,        sizeof(float)*size);
	float* devPBest;       cudaMalloc((void**)&devPBest,      sizeof(float)*size);
	float* devGBest;       cudaMalloc((void**)&devGBest,      sizeof(float)*num_dimensions);
	float* tempParticle1;  cudaMalloc((void**)&tempParticle1, sizeof(float)*num_dimensions);
	float* tempParticle2;  cudaMalloc((void**)&tempParticle2, sizeof(float)*num_dimensions);
	float  temp[num_dimensions];

	// Thread & Block number
	int threadsNum = 32;
	int blocksNum = ceil(size / threadsNum);

	// Copy particle datas from host to device. Copy in GPU memory the data from the host
	cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVel, velocities, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devGBest, gBest, sizeof(float) * num_dimensions, cudaMemcpyHostToDevice);

	// PSO main function
	for(int iter=0; iter<max_iter; ++iter){
		kernelUpdateParticle<<<blocksNum, threadsNum>>>(num_dimensions,num_particles, devPos,devVel,devPBest,devGBest, getRandomClamped(),getRandomClamped());
		kernelUpdatePBest   <<<blocksNum, threadsNum>>>(num_dimensions,num_particles, devPos,devPBest,devGBest, tempParticle1,tempParticle2);
		cudaMemcpy(pBests, devPBest, sizeof(float) * num_particles * num_dimensions, cudaMemcpyDeviceToHost);

		for(int i=0; i<size; i+=num_dimensions){
			for(int k=0; k<num_dimensions; ++k)
				temp[k]=pBests[i+k];
			if(fitness_function(num_dimensions, temp) < fitness_function(num_dimensions, gBest))
				for(int k=0; k < num_dimensions; ++k)  gBest[k] = temp[k];
		}

		cudaMemcpy(devGBest, gBest, sizeof(float) * num_dimensions,cudaMemcpyHostToDevice);
	}

	cudaMemcpy(positions,  devPos,   sizeof(float)*size,           cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities, devVel,   sizeof(float)*size,           cudaMemcpyDeviceToHost);
	cudaMemcpy(pBests,     devPBest, sizeof(float)*size,           cudaMemcpyDeviceToHost);
	cudaMemcpy(gBest,      devGBest, sizeof(float)*num_dimensions, cudaMemcpyDeviceToHost);
	cudaFree(devPos);
	cudaFree(devVel);
	cudaFree(devPBest);
	cudaFree(devGBest);
	cudaFree(tempParticle1);
	cudaFree(tempParticle2);
}

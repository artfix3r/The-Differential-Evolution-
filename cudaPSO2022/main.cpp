#include "kernel.h"

// @param nruns: number of runs, in order to calculate the mean and standard deviation after @nruns
// @param D:     dimensionality of the problem
// @param P:     population size
// @param G:     number of generations
void run(int nruns, int DN,int PN,int Di,int Pi, int D,int P,int G){
	printf("  {\n");
	printf("    \"dim\": %d,\n",  D);  // printf("\x1b[92m%s  \x1b[0mpop \x1b[31m%'5d  \x1b[0mdim \x1b[32m%'3d  \x1b[0mgen \x1b[34m%'5d\x1b[0m\n", __func__,P,D,G);
	printf("    \"pop\": %d,\n",  P);
	printf("    \"gen\": %d,\n",  G);
	printf("    \"rnuns\": %d,\n", nruns);

	float* positions  = (float*)malloc(sizeof(float)*P*D);
	float* velocities = (float*)malloc(sizeof(float)*P*D);
	float* pBests     = (float*)malloc(sizeof(float)*P*D);
	float* gBest      = (float*)malloc(sizeof(float)*D);

	float* x_mean     = (float*)malloc(sizeof(float)*D);
	float* x_std      = (float*)malloc(sizeof(float)*D);
	float* x_sum      = (float*)calloc(D,sizeof(float));  // running sum            for each coordinate
	float* x_sumsq    = (float*)calloc(D,sizeof(float));  // running sum of squares for each coordinate
	float  y_mean;
	float  y_std;
	float  y_sum      = 0;  // running sum            for the minimum function value
	float  y_sumsq    = 0;  // running sum of squares for the minimum function value

	// ----------------------------------------------------------------
	for(int run=0; run<nruns; ++run){
		for(int i=0; i<P*D; ++i){
			positions[i]  = getRandom(START_RANGE_MIN, START_RANGE_MAX);
			pBests[i]     = positions[i];
			velocities[i] = 0;
		}
		for(int k=0; k<D; ++k)  gBest[k] = pBests[k];

		clock_t t0 = clock();
		cuda_pso(D,P,G, positions,velocities,pBests,gBest);   // the vector that minimizes the objective function should be @gBest
		clock_t t1 = clock();
		// printf("    time %.3f  min %6.3f  res", (double)(t1-t0)/CLOCKS_PER_SEC, fitness_function(D,gBest));  for(int k=0; k<D; ++k) printf(" %6.3f",gBest[k]);  putchar(0x0a);

		for(int k=0; k<D; ++k){
			float xk    = gBest[k];
			x_sum[k]   += xk;
			x_sumsq[k] += xk*xk;
			x_mean[k]   = x_sum[k]  /(run+1);
			x_std[k]    = x_sumsq[k]/(run+1) - x_mean[k]*x_mean[k];  // printf("gBest ");  for(int k=0; k<D; ++k) printf(" %6.3f", xk);  putchar(0x0a);
		}
		float y  = fitness_function(D,gBest);
		y_sum   += y;
		y_sumsq += y*y;
		y_mean   = y_sum  /(run+1);
		y_std    = y_sumsq/(run+1) - y_mean*y_mean;  // printf("gBest ");  for(int k=0; k<D; ++k) printf(" %6.3f", gBest);  putchar(0x0a);
	}

	printf("    \"x_mean\":[");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_mean[i]);  printf("%6.3f", x_mean[D-1]); printf("],\n");
	printf("    \"x_std\": [");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_std[i]);   printf("%6.3f", x_std[D-1]);  printf("],\n");
	printf("    \"y_mean\": %6.3f,\n", y_mean);
	printf("    \"y_std\":  %6.3f\n",  y_std);
	printf("  }");
	if(!(Di==DN-1 && Pi==PN-1))  printf(",");
	putchar(0x0a);

	// ----------------------------------------------------------------
	free(positions);
	free(velocities);
	free(pBests);
	free(gBest);
	free(x_mean);
	free(x_std);
	free(x_sum);
	free(x_sumsq);

	// printf("    \"result_mean\":[");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_mean[i]);  printf("%6.3f", x_mean[D-1]); printf("],\n");
	// printf("    \"result_std\": [");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_std[i]);   printf("%6.3f", x_std[D-1]);  printf("]\n");
	// printf("  }");
	// if(!(Di==DN-1 && Pi==PN-1))  printf(",");
	// putchar(0x0a);
}

int main(int nargs, char* args[]){
	srand((unsigned)time(NULL));

	int NRUNS = 25;  // number of runs for each configuration, in order to calculate the mean and standard deviation
	// int D[]   = {2, 5,10};  // dimensionality of the problem
	// int P[]   = {5,10,50};  // population size
	int D[]   = {10, 50,100};  // dimensionality of the problem
	int P[]   = {50,100,500};  // population size

	puts("[");
	for(int i=0; i<arridim(D); ++i)
		for(int j=0; j<arridim(P); ++j)
			run(NRUNS, arridim(D),arridim(P),i,j, D[i],P[j],104*D[i]);
	puts("]");
	return 0;
}

// 	for(int run=0; run<nruns; ++run){
// 		DifferentialEvolution minimizer(P,G,D, CR,F, minBounds,maxBounds);  // create the minimizer with a popsize of P, G generations, D dimensions
// 		std::vector<float> z = minimizer.fmin(d_x);  // get the @x from the minimizer, where @x is the vector that maximizes the cost function?
// 		for(int i=0; i<D; ++i){
// 			x_sum[i]   += z[i];
// 			x_sumsq[i] += z[i]*z[i];
// 			x_mean[i] = x_sum[i]  /(run+1);
// 			x_std[i]  = x_sumsq[i]/(run+1) - x_mean[i]*x_mean[i];  // printf("result ");  for(int i=0; i<D; ++i) printf(" %6.3f", z[i]);  putchar(0x0a);
// 		}
// 	}
// 	printf("    \"result_mean\":[");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_mean[i]);  printf("%6.3f", x_mean[D-1]); printf("],\n");
// 	printf("    \"result_std\": [");  for(int i=0; i<D-1; ++i) printf("%6.3f,", x_std[i]);   printf("%6.3f", x_std[D-1]);  printf("]\n");
// 	printf("  }");
// 	if(!(Di==DN-1 && Pi==PN-1))  printf(",");
// 	putchar(0x0a);
// }

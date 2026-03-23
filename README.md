# code

An optimization algorithm takes a scalar function `f:R^d --> R` (that maps d-dimensional vectors to scalars) and tries to find the vector `x` that achieves the minimum possible value `y` (or `f(x)`).

The function `f` is also called the `cost function` or the `loss function`, and it's typically minimized. (Although one can solve an equivalent optimization problem by maximizing `-f`).

## DE

The Differential Evolution (DE) code iterates over all given dimensionalities and population sizes and, for each dimension `D` and population `P`, it runs the DE algorithm `N` times by calling `DifferentialEvolution minimizer.fmin()`.

Each time the optimizer `DifferentialEvolution minimizer.fmin()` runs, it returns the vector `x` that the minimum value `y` the value of the function `f`.

As we run the optimizer `DifferentialEvolution minimizer.fmin()` `N` times we record the `x` and `y` values that are returned each run, and compute the mean and standard deviation for them.  
(Since `x` is a `d`-dimensional vector, it contains `d` values, and we compute the mean and standard deviation for each of those `d` values independently, so that have a `d`-dimensional array of means and a `d`-dimensional array of standard deviations for `x`.)  
The mean and standard deviation of the `x` values are stored in the `x_mean` and `x_std` variables, respectively.  
The mean and standard deviation of the `y` value  is  stored in the `y_mean` and `y_std` variables, respectively.  

The `DifferentialEvolution minimizer.fmin()` function (defined in `DifferentialEvolution.cpp`) calls the function `differentialEvolution()` (defined on `DifferentialEvolutionGPU.cu`).
The `differentialEvolution()` function is a CPU function (aka. "host" function) that serves an the entry point into CUDA kernel launches. It:

0) Calls the GPU kernel ("global" function) `generateRandomVectorAndInit()` to randomly initialize the parameters (ie. an `x` vector for each of the population members), and compute the cost function `f` on those random `x` vectors.  
1) Calls the GPU kernel ("global" function) `evolutionKernel()` `maxGenerations` times.  
2) Afterwards, on the CPU, it finds the index of the best agent (where the "best agent" is the agent that found the lowest value for the function `f`).  

The `evolutionKerne()` CUDA kernel is what runs one step/iteration of the DE algorithm, and it works roughly as follows, according to my understanding.

0) For each agent `i` in the population, choose 3 random agents `a`, `b`, and `c`, distinct from  `p`,  and compute a new agent `trial` based on the agents `i`, `a`, `b`, `c`, `j`, global parameters `CR` and `F`, and a random number `r`, as follows:  
	- if `r<CR`, the `trial` becomes `a + F*(b-c)`  
	- else, `trial` becomes `i`.  
1) Evaluate the cost function `f` at the new agent `trial` to get `f(trial)`.  
2) If `f(trial)` is smaller then `f(i)`, then use `trial` instead of `i` for the next iteration.  

The code was also cleaned up: inconsequent spaces were removed, and the code was reformatted somewhat to be easier to read.

## PSO

The Particle Swarm Optimization (PSO) code iterates over all given dimensionalities and population sizes and, for each dimension `D` and population `P`, it runs the PSO algorithm `N` times by calling `cuda_pso`.

Each time the optimizer `cuda_pso()` runs, it returns the vector `x` that the minimum value `y` the value of the function `f`.

As we run the optimizer `cuda_pso()` `N` times we record the `x` and `y` values that are returned each run, and compute the mean and standard deviation for them.  
(Since `x` is a `d`-dimensional vector, it contains `d` values, and we compute the mean and standard deviation for each of those `d` values independently, so that have a `d`-dimensional array of means and a `d`-dimensional array of standard deviations for `x`.)  
The mean and standard deviation of the `x` values are stored in the `x_mean` and `x_std` variables, respectively.  
The mean and standard deviation of the `y` value  is  stored in the `y_mean` and `y_std` variables, respectively.  

The `cuda_pso()` function (defined in `kernel.cu`) is a CPU function (aka. "host" function) that serves an the entry point into CUDA kernel launches. It:

0) allocates memory for the various GPU ("device") memory buffers, and for 1 CPU memory buffer.  
1) Runs for `max_iter` iterations, and, on each iteration, it performs one step of the PSO algorithm.  
2) On each iteration it:  
	2.0) launches the GPU kernel `kernel_particles()`,  
	2.0) launches the GPU kernel `kernel_pbests()`,  
	2.1) finds the best performing agent on the CPU (this is by far the slowest part, taking `~50%` to `~97%` of the PSO running time (depending on problem size), according to my tests.  

The `kernel_particles()` GPU kernel does the following (based on the parameters `OMEGA`, `c1`, `c1`, and 2 random numbers `rp` and `rg`), according to my understanding:

0) updates the velocity of each agent `i` in the population according to the PSO update formula: `velocity[i] = OMEGA*velocity[i] +  c1*rp*(pbest[i]-position[i]) + c2*rg*(gbest-position[i])`, where `position[i]`, `velocity[i]`, `pbest[i]`, and `gbest` are `d`-dimensional vectors,  
1) updates the position of each agent `i` in the population according to the (discrete) calculus update formula: `position = position + velocity`.  

The `kernel_pbests()` GPU kernel does the following, according to my understanding:

0) stores the `positions` in `tmp_particle0`,  
1) stores the `pbests` in `tmp_particle1`,  
2) updates `pbest` by `position` for the agents for which `position` yields a lower value of the cost function `f`.  

After those 2 GPU kernels have finished running the CPU finds the best `pbest` (which is a list of `d`-dimensional agent), and stores in it `gbest` (which is a single `d`-dimensional agent).

The code was also cleaned up: inconsequent spaces were removed, and the code was reformatted somewhat to be easier to read.
The number of files was reduced.
The number of compilation units was reduce to 2. Removed a duplicate of the function `fitness_function()` by using adding `__host__` directive to `__device__ fitness_function()`.
Modified the code to reparametrize the PSO function over the dimensionality, population size, max generations, etc. (before, these were global, hard-coded variables).

# summary activities

- Cleaned up the code somewhat.
- Added Rastrigin's function to the DE code, as a device function.
- Ran the Differential Evolution (DE) and the PSO (Particle Swarm Optimization) algorithms over various dimension and population parameters, for multiple runs.
- The 2 CUDA programs (for DE and PSO) output JSON, which the Python programs htest.py can then read to compute the statistical tests.
- For PSO, I used Levy 3-dimensional, Shifted Rastigrin's Function, Shifted Rosenbrock's Function, Shifted Griewank's Function, Shifted Sphere's Function.
- Performed Wilcoxon Signed-Rank Test and Kruskal-Wallis H Test.
- Added the quadratic cost function to PSO.
- Attempted to naively fuse the sequential step in PSO into a kernel. The execution time gets cut by ~40% to ~97% (depending on problem size), but the results are incorrect. A parallel max reduce would be needed to get correct results from this optimization, I propose.

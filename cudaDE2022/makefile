all: programDE

programDE: main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu
	nvcc  main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu  -o programDE  -use_fast_math -O3

clean:
	rm programDE

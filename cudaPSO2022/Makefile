CXX=gcc

CUDA_INSTALL_PATH=/usr/local/cuda-11.2
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64

all: pso

pso: main.cpp kernel.h Makefile  kernel.o
	$(CXX) $(CFLAGS) -c main.cpp -o main.o
	$(CXX) $(LDFLAGS)  main.o kernel.o -o pso -lcudart -lm -lstdc++

kernel.o: kernel.cu kernel.h Makefile
	nvcc -c kernel.cu -o kernel.o  -use_fast_math -O3

clean:
	rm -f *.o pso

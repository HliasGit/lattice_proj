# Variables
CC = g++
BIN_FOL = bin
NVCC = nvcc
FILE_PAR = gen_parallel_algo.cu
OUT_PAR = parallel_mio
FILE_SEQ = sequential.cpp
OUT_SEQ = sequential
FILE_FIX = fixed_point.cu
OUT_FIX = fixpoint

# Targets
all: make sequential gen_parallel_algo fixed_point

make: 
	mkdir -p bin/

sequential: $(FILE_SEQ)
	$(CC) $(FILE_SEQ) -o $(BIN_FOL)/$(OUT_SEQ)

gen_parallel_algo: $(FILE_PAR)
	$(NVCC) $(FILE_PAR) -o $(BIN_FOL)/$(OUT_PAR)

fixed_point: $(FILE_FIX)
	$(NVCC) $(FILE_FIX) -o $(BIN_FOL)/$(OUT_FIX)


clean:
	rm -rf $(BIN_FOL)/
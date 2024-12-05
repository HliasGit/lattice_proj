# Variables
BIN_FOL = bin
G++ = g++
NVCC = nvcc
FILE_SEQ = fixpoint_sequential.cpp
OUT_SEQ = fixpoint_sequential
FILE_PAR = fixpoint_parallel.cu
OUT_PAR = fixpoint_parallel

# Targets
all: make sequential parallel

make:
	mkdir -p bin/

sequential: $(FILE_SEQ)
	$(G++) $(FILE_SEQ) -o $(BIN_FOL)/$(OUT_SEQ)

parallel: $(FILE_PAR)
	$(NVCC) $(FILE_PAR) -o $(BIN_FOL)/$(OUT_PAR)

clean:
	rm -rf $(BIN_FOL)/
# Variables
CC = g++
BIN_FOL = bin
NVCC = nvcc
FILE_SEQ = nqueens.cpp
OUT_SEQ = nqueens

# Targets
all: make sequential

make:
	mkdir -p bin/

sequential: $(FILE_SEQ)
	$(CC) $(FILE_SEQ) -o $(BIN_FOL)/$(OUT_SEQ)

clean:
	rm -rf $(BIN_FOL)/
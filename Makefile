# I want a makefile to compile my project
# Compiler: g++
# Flags: -Wall -Wextra -Werror -std=c++11
# Files: main.cpp, functions.cpp, functions.h
# Output: main

# Variables
CC = g++
NVCC = nvcc
FILE_PAR = gen_parallel_algo.cu
OUT_PAR = parallel_mio
FILE_SEQ = sequential.cpp
OUT_SEQ = sequential
FILE_FIX = fixed_point.cu
OUT_FIX = fixpoint
FILE_EXE2 = exercise_2.cu
OUT_EXE2 = exe2

# Targets
all: sequential parallel_mio fixpoint

sequential: $(FILE_SEQ)
	$(CC) $(FILE_SEQ) -o $(OUT_SEQ)

parallel_mio: $(FILE_PAR)
	$(NVCC) $(FILE_PAR) -o $(OUT_PAR)

fixpoint: $(FILE_FIX)
	$(NVCC) $(FILE_FIX) -o $(OUT_FIX)

exercise_2: $(FILE_EXE2)
	$(NVCC) $(FILE_EXE2) -o $(OUT_EXE2)

clean:
	rm -f $(OUT_SEQ) $(OUT_PAR) $(OUT_FIX)
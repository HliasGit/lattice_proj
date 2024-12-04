# I want a makefile to compile my project
# Compiler: g++
# Flags: -Wall -Wextra -Werror -std=c++11
# Files: main.cpp, functions.cpp, functions.h
# Output: main

# Variables
CC = g++
BIN_FOL = bin
NVCC = nvcc
FILE_SEQ = exercise.cpp
OUT_SEQ = fixpoint

# Targets
all: sequential

sequential: $(FILE_SEQ)
	$(CC) $(FILE_SEQ) -o $(BIN_FOL)/$(OUT_SEQ)

clean:
	rm -rf $(BIN_FOL)/
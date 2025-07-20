# --- Compiler ---
CC = gcc
NVCC = nvcc

# --- Flags ---
CFLAGS = -c -g -fopenmp -I$(HEADER_DIR)
NVFLAGS = -c -O4 -arch=sm_89 -I$(HEADER_DIR)
LDFLAGS = -lm -lGL -lGLU -lglut -lpng -fopenmp -lcufft -lcudart

# --- Directories ---
SRC_DIR = src
HEADER_DIR = headers
BIN_DIR = bin
OBJ_DIR = $(BIN_DIR)/obj

# --- Source files ---
SRC_C = $(wildcard *.c) $(wildcard $(SRC_DIR)/*.c)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)

# --- Object files ---
OBJ_C = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRC_C)))
OBJ_CU = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(notdir $(SRC_CU)))

OBJ = $(OBJ_C) $(OBJ_CU)

# --- Target executable ---
TARGET = $(BIN_DIR)/main

# --- Default rule ---
all: $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

# --- Compilation rules ---
$(OBJ_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -Ofast $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -Ofast $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# --- Specific dependencies (optionnel si besoin précis) ---
$(OBJ_DIR)/png_util.o : $(SRC_DIR)/png_util.c $(HEADER_DIR)/png_util.h
$(OBJ_DIR)/life.o : $(SRC_DIR)/life.c $(HEADER_DIR)/life.h $(OBJ_DIR)/convolve.o

# --- Utility rules ---
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean

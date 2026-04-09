###############################################################################
# GPU Image Deduplicator - Makefile
###############################################################################

# Compiler
NVCC = nvcc

# Directories
SRC_DIR = src
BIN_DIR = bin

# Target
TARGET = $(BIN_DIR)/gpu_dedup

# Source files
SOURCES = $(SRC_DIR)/main.cu

# Compiler flags
NVCC_FLAGS = -O2 -std=c++17
# Include source dir for stb headers
NVCC_FLAGS += -I$(SRC_DIR)
# Suppress stb warnings
NVCC_FLAGS += -Xcompiler "-w"

# Architecture flags (adjust for your GPU)
# Default: generate for common architectures
ARCH_FLAGS = -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

run: $(TARGET)
	./run.sh

help:
	@echo "Usage:"
	@echo "  make          - Build the project"
	@echo "  make run      - Build and run with sample data"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"

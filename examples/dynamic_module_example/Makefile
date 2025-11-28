# Simple Makefile for quick building without CMake

CXX = hipcc
CXXFLAGS = -fgpu-rdc -O2 -I../../include
LDFLAGS = -L../../build -lrocshmem -fgpu-rdc

all: dynamic_load_example kernel.hsaco

# Compile main executable
dynamic_load_example: dynamic_load_example.cpp
	@echo "=== Building main executable ==="
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✓ Built $@"

# Compile kernel to dynamic module
kernel.hsaco: kernel.hip
	@echo "=== Building dynamic module ==="
	@chmod +x compile_module.sh
	./compile_module.sh
	@echo "✓ Built $@"

# Run with single PE
run: all
	@echo "=== Running example (single PE) ==="
	./dynamic_load_example

# Run with multiple PEs (requires MPI)
run-mpi: all
	@echo "=== Running example (2 PEs) ==="
	mpirun -np 2 ./dynamic_load_example

clean:
	rm -f dynamic_load_example kernel.hsaco kernel.bc kernel_linked.bc

.PHONY: all run run-mpi clean


MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

GPP:= g++
#GPP:= /sw/gcc/11.2.0/bin/g++
ifeq ($(CUDA_HOME),)
	CUDA_HOME:= $(shell which nvcc | rev | cut -d'/' -f3- | rev)
endif
ifeq ($(ROCM_HOME),)
	ROCM_HOME:= $(shell which hipcc | rev | cut -d'/' -f3- | rev)
endif

ifndef CUDA_VERSION
$(warning WARNING: CUDA_VERSION not set. Call make with CUDA string, for example: make cuda11x CUDA_VERSION=115 or make cpuonly CUDA_VERSION=CPU)
CUDA_VERSION:=
endif

ifdef GITHUB_ACTIONS
OUTDIR:= BuildArtifacts
else
OUTDIR:= bitsandbytes
endif



NVCC := $(CUDA_HOME)/bin/nvcc

###########################################

CSRC := $(ROOT_DIR)/csrc
BUILD_DIR:= $(ROOT_DIR)/build

FILES_CUDA := $(CSRC)/ops.cu $(CSRC)/kernels.cu
FILES_CPP := $(CSRC)/common.cpp $(CSRC)/cpu_ops.cpp $(CSRC)/pythonInterface.c

INCLUDE :=  -I $(CUDA_HOME)/include -I $(ROOT_DIR)/csrc -I $(CONDA_PREFIX)/include -I $(ROOT_DIR)/include
LIB := -L $(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lcusparse -L $(CONDA_PREFIX)/lib

# NVIDIA NVCC compilation flags
COMPUTE_NOCUBLAST := -gencode arch=compute_50,code=sm_50 # Maxwell
COMPUTE_NOCUBLAST += -gencode arch=compute_52,code=sm_52 # Maxwell
COMPUTE_NOCUBLAST += -gencode arch=compute_60,code=sm_60 # Pascal
COMPUTE_NOCUBLAST += -gencode arch=compute_61,code=sm_61 # Pascal
COMPUTE_NOCUBLAST += -gencode arch=compute_70,code=sm_70 # Volta

CC_KEPLER := -gencode arch=compute_35,code=sm_35 # Kepler
CC_KEPLER += -gencode arch=compute_37,code=sm_37 # Kepler

# Later versions of CUDA support the new architectures
CC_CUDA11x := -gencode arch=compute_75,code=sm_75
CC_CUDA11x += -gencode arch=compute_80,code=sm_80
CC_CUDA11x += -gencode arch=compute_86,code=[compute_86,sm_86]

CC_CUDA118 := -gencode arch=compute_75,code=sm_75
CC_CUDA118 += -gencode arch=compute_80,code=sm_80
CC_CUDA118 += -gencode arch=compute_86,code=sm_86
CC_CUDA118 += -gencode arch=compute_89,code=sm_89
CC_CUDA118 += -gencode arch=compute_90,code=[compute_90,sm_90]

CC_ROCM := --offload-arch=gfx900 --offload-arch=gfx90agfx904 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx940 --offload-arch=gfx941 --offload-arch=gfx942 --offload-arch=gfx1010 --offload-arch=gfx1011 --offload-arch=gfx1012 --offload-arch=gfx1013 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1034


all: $(BUILD_DIR) env
	$(NVCC) $(CC_CUDA11x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_CUDA11x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cuda11x_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA11x) $(CC_KEPLER) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA11x) $(CC_KEPLER) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda118_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA118) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA118) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda12x_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA118) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_NOCUBLAST) $(CC_CUDA118) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda11x: $(BUILD_DIR) env
	$(NVCC) $(CC_CUDA11x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_CUDA11x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cuda118: $(BUILD_DIR) env
	$(NVCC) $(CC_CUDA118) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_CUDA118) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cuda12x: $(BUILD_DIR) env
	$(NVCC) $(CC_CUDA118) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_CUDA118) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

hip: $(BUILD_DIR) env
	$(HIPCC) -std=c++14 $(CC_ROCM) -fPIC -c -DNO_CUBLASLT $(INCLUDE) $(LIB) $(CSRC)/ops.cu -o $(BUILD_DIR)/ops.o
	$(HIPCC) -std=c++14 $(CC_ROCM) -fPIC -c -DNO_CUBLASLT $(INCLUDE) $(LIB) $(CSRC)/kernels.cu -o $(BUILD_DIR)/kernels.o
	$(GPP) -std=c++14 -D__HIP_PLATFORM_AMD__ -DBUILD_CUDA -DNO_CUBLASLT -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_hip_nohipblaslt.so $(LIB)

cpuonly: $(BUILD_DIR) env
	$(GPP) -std=c++14 -shared -fPIC -I $(ROOT_DIR)/csrc -I $(ROOT_DIR)/include $(FILES_CPP) -o ./$(OUTDIR)/libbitsandbytes_cpu.so

env:
	@echo "ENVIRONMENT"
	@echo "============================"
	@echo "CUDA_VERSION: $(CUDA_VERSION)"
	@echo "============================"
	@echo "NVCC path: $(NVCC)"
	@echo "HIPCC path: $(HIPCC)"
	@echo "GPP path: $(GPP) VERSION: `$(GPP) --version | head -n 1`"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "ROCM_HOME: $(ROCM_HOME)"
	@echo "CONDA_PREFIX: $(CONDA_PREFIX)"
	@echo "PATH: $(PATH)"
	@echo "LD_LIBRARY_PATH: $(LD_LIBRARY_PATH)"
	@echo "============================"

$(BUILD_DIR):
	mkdir -p build
	mkdir -p dependencies

$(ROOT_DIR)/dependencies/cub:
	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub
	cd dependencies/cub; git checkout 1.11.0

clean:
	rm build/*

cleaneggs:
	rm -rf *.egg*

cleanlibs:
	rm ./bitsandbytes/libbitsandbytes*.so

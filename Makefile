GPU_TARGET=H100

# Compiler
NVCC?=nvcc
# NVCC?=/usr/local/cuda-12.6/bin/nvcc

# Conditional setup based on the target GPU
# NVCCFLAGS=-DNDEBUG -G -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -Isubmodules/ThunderKittens/include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static # H100
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -Isubmodules/ThunderKittens/include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static # H100

# 
TARGET=a.out # H100

# SRC=test_asyncload_wg.cu # H100
# SRC=test_asyncload.cu # H100
# SRC=test_fullattn.cu # H100
SRC=test_async_gather_attn.cu # H100
# SRC=test_async_gather_attn_topk.cu # H100


# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)


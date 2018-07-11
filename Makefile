CFLAGS := -DOPENCV_TRAITS_ENABLE_DEPRECATED  -Dpbcvt_EXPORTS  -I/usr/include/python3.5m/ `pkg-config --cflags opencv`  #-fpermissive

LDFLAGS := -fopenmp -ftree-vectorize -lboost_python3 -lfftw3f -lfftw3f_omp `pkg-config --libs opencv`


ifdef CUDA
LDFLAGS := $(LDFLAGS) -L/usr/local/cuda/lib64/  -lcudart -lcufft
CFLAGS := $(CFLAGS) -I/usr/local/cuda/include/ -DCUDA #-DJETSON -I/usr/local/cuda-8.0/targets/aarch64-linux/include
endif


ifdef PROF
LDFLAGS := $(LDFLAGS) -lprofiler
CFLAGS := $(CFLAGS) -DPROFILER
endif

GCC_CFLAGS := $(CFLAGS) -fopenmp -fPIC -O3 -ggdb --std=gnu++11
CUDA_CFLAGS := $(CFLAGS) -O3 --std=c++11 -Xcompiler -fPIC -Xptxas="-v" --gpu-architecture=sm_52



# LDFLAGS := -fopenmp -lboost_python3 -lfftw3f -lfftw3f_omp `pkg-config --libs opencv`

.PHONY: all
.PHONY: clean

SRCS = $(wildcard cpp/*.cpp)

ifdef CUDA
CUDA_SRCS = $(wildcard cpp/*.cu)
else
CUDA_SRCS =
endif

PROGS = $(patsubst cpp/%.cpp,build/%.o,$(SRCS))

CUDA_PROGS = $(patsubst cpp/%.cu,build/%.o,$(CUDA_SRCS))
CUDA_PTXS = $(patsubst cpp/%.cu,ptx/%.ptx,$(CUDA_SRCS))
EXC_PROGS = $(filter-out build/py%.o, $(PROGS))

.PHONY: all
.PHONY: clear
.PHONY: ptx


all: main #mosse.so

clean:
	rm -f build/*.o

ptx: $(CUDA_PTXS)

#mosse.so: Makefile $(PROGS) $(CUDA_PROGS)
#	g++ -o $@ -shared  $(PROGS) $(CUDA_PROGS)  $(LDFLAGS)

main: Makefile $(EXC_PROGS) $(CUDA_PROGS) build/main.o
	g++ -o $@ build/main.o $(EXC_PROGS) $(CUDA_PROGS) $(LDFLAGS)

build/main.o: Makefile main.cpp
	g++ -o $@ -c main.cpp $(GCC_CFLAGS)

build/%.o: cpp/%.cpp
	g++ -c $< $(GCC_CFLAGS) -o $@

ifdef CUDA
build/%.o: cpp/%.cu
	nvcc -c $< $(CUDA_CFLAGS) -o $@

ptx/%.ptx: cpp/%.cu
	nvcc -ptx $< $(CUDA_CFLAGS) -o $@
endif

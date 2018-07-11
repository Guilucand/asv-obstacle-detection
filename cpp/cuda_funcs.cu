#include "cuda_funcs.h"
#include <iostream>
#include <cuComplex.h>
#include <limits.h>

using namespace cuda_funcs;
using namespace std;
using namespace cv;


/*
x -> Number of threads foreach row [8]
numIterations -> Number of iterations foreach thread [4]
y -> Number of rows, x threads per row [32]
x * y * numIterations -> Number of pixels [1024]
z -> Number of patches [4] => total threads per block => 1024

row_stride -> row stride [32]
stride -> patch stride [1024]

bx -> total number of patches batches [count/4 rounded up]

*/

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

template <class T>
__inline__ __device__
void warpReduceSumTwo(T &val1, T &val2) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val1 += __shfl_down(val1, offset);
    val2 += __shfl_down(val2, offset);
  }
  return;
}

template <class T>
__inline__ __device__
void warpReduceMaxIdx(T &maxm, int idx[2]) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    T rvalue = __shfl_down(maxm, offset);
    maxm = max(maxm, rvalue);
    idx[maxm==rvalue] = __shfl_down(idx[1], offset);
  }
  return;
}

// Texture holding the whole image for preprocessing
static texture<float, 2, cudaReadModeElementType> patchTex;

static __global__ void preprocess_kernel(const Point *locations,
                                  float *mat,
                                  float *window, int window_row_stride,
                                  int row_stride, int stride,
                                  float2 xtr,
                                  float2 ytr,
                                  int count) {

  const int IterationsPerThread = 4;
  const int NumberOfColumns = blockDim.x;
  const int RowThreadBlocksCount = blockDim.y;
  const int SubPatchCount = blockDim.z;

  const int ThreadsPerPatch = RowThreadBlocksCount * NumberOfColumns;

  const int TotalPixelsPerPatch = ThreadsPerPatch * IterationsPerThread;

  const int ThreadPatchStride = row_stride * RowThreadBlocksCount;
  const int ThreadRowStride = RowThreadBlocksCount;

  const int WarpsPerPatch = ThreadsPerPatch / warpSize;

  const int subColIndex = threadIdx.x;
  const int rowIndex = threadIdx.y;
  const int subPatchIndex = threadIdx.z;
  const int patchBlockIndex = blockIdx.x;

  // single shared memory
  extern __shared__ float shared[];

  float logsRegs[IterationsPerThread];

  // Compute accumulators start address (skip mean and std)
  float * __restrict__ accum = shared + (SubPatchCount * 2);

  // Find current patch index
  int patchIdx =  subPatchIndex + patchBlockIndex * SubPatchCount;

  if (patchIdx >= count) // Exit if outside patches count
    return;

  // Find start pointer to the current patch
  float * __restrict__ current_patch = mat + (patchIdx * stride);

  const int patch_thread_index = subColIndex + (rowIndex * NumberOfColumns);
  const int warpIndex = patch_thread_index/warpSize;

  // Compute current thread shared index => [0-256] first patch [256-512] second patch etc...
  const int patch_accum_index = (subPatchIndex * WarpsPerPatch) * 2;
  float * __restrict__ __mean_accum = accum + patch_accum_index + warpIndex;
  float * __restrict__ __sqmean_accum = accum + patch_accum_index + warpIndex + WarpsPerPatch;

  const Point &coords = locations[patchIdx];


  // Set accumulators to 0
  float mean_accum = 0;
  float sqmean_accum = 0;

  int lOffset = 0;
  int rowOffset = rowIndex;

  // Normalize current patch and save it to logs shared variable
  while (lOffset < IterationsPerThread) {

    const int cx =  subColIndex; // Column index *NEVER* changes
    const int cy = rowOffset;

    float2 transPixel = make_float2(coords.x + xtr.x * cx + ytr.x * cy,
                                    coords.y + xtr.y * cx + ytr.y * cy);

    float pixel = tex2D(patchTex, transPixel.x + 0.5f, transPixel.y + 0.5f);

    const float mult = 8192.0f;
    float tmpVal = logsRegs[lOffset] = max(0.0f, logf(((pixel)*(mult/256.0f)) + 1.0f));// + 0.0001f);//current_row[localOffset];//
    mean_accum += tmpVal;
    sqmean_accum += tmpVal*tmpVal;
    rowOffset += ThreadRowStride;
    lOffset++;
  }
  __syncthreads();

  int laneidx = lane_id();
  warpReduceSumTwo(mean_accum, sqmean_accum);
  if (!laneidx) {
    __mean_accum[0] = mean_accum;
    __sqmean_accum[0] = sqmean_accum;
  }
  __syncthreads();

  // First warp
  if (patch_thread_index < WarpsPerPatch) {
    float gmean_accum = accum[patch_accum_index + patch_thread_index];
    float gsqmean_accum = accum[patch_accum_index + patch_thread_index + WarpsPerPatch];
    warpReduceSumTwo(gmean_accum, gsqmean_accum);

    if (!patch_thread_index) {
      // Array of 2 values, mean and std
      float2 &meanStandardDeviation = ((float2*)shared)[subPatchIndex];

      float mean = meanStandardDeviation.x = mean_accum/TotalPixelsPerPatch;
      // Compute std
      float std = sqrt(sqmean_accum/TotalPixelsPerPatch - mean*mean);
      const float eps = 1e-5;
      float mult = 1.0f/(std+eps);
      meanStandardDeviation.y = mult;
    }
  }
  __syncthreads();

  float2 meanStd = ((float2*)shared)[subPatchIndex];

  int patchOffset = subColIndex + (rowIndex * row_stride);
  int windowOffset = subColIndex + (rowIndex * window_row_stride);
  for (int regIdx = 0; regIdx < IterationsPerThread; regIdx++) {
    current_patch[patchOffset] = (logsRegs[regIdx] - meanStd.x) * meanStd.y * window[windowOffset];
    patchOffset += ThreadPatchStride;
    windowOffset += ThreadRowStride * window_row_stride;
  }
}

cudaArray *cuArray;
Point *cudaPoints;
int max_size;

void cuda_funcs::preprocess(vector<MosseWindow*> const& windows, Size winSize, Matx33f trMat, Mat const& patch, Mat &first, Mat &window) {

  if (!cuArray) {
    // Allocate array and bind to texture
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaMallocArray(&cuArray,
                    &channelDesc,
                    patch.cols,
                    patch.rows);

    cudaBindTextureToArray(patchTex, cuArray, channelDesc);
  }

  // Allocate points array if not allocated
  if (!cudaPoints) {
    cudaMallocManaged((void**)&cudaPoints, max_size * sizeof(Point));
  }

  // Transform points to window coordinates
  for (int i = 0; i < windows.size(); i++) {
    auto point = Matx13f(windows[i]->position.x - winSize.width/2,
                          windows[i]->position.y - winSize.height/2, 1) * trMat;
    cudaPoints[windows[i]->index] = Point(point(0, 0), point(0, 1));
  }

  cudaMemcpy2DToArray(cuArray,
                      0,
                      0,
                      patch.data,
                      patch.step,
                      patch.cols * sizeof(float),
                      patch.rows,
                      cudaMemcpyHostToDevice);

  // Set texture parameters
  patchTex.addressMode[0] = cudaAddressModeBorder;
  patchTex.addressMode[1] = cudaAddressModeBorder;
  patchTex.filterMode = cudaFilterModePoint;
  patchTex.normalized = false;    // access with normalized texture coordinates

  int count = max_size;//windows.size();
  dim3 blockSize(32, 8, 4);
  int numThreads = 1024;//blockSize.x * blockSize.y * blockSize.z;
  dim3 gridSize((count + 3) / 4, 1, 1);

  int row_stride = first.step/sizeof(float);
  // printf("Row stride: %d cols: %d\n", row_stride, first.cols);

  int memBytes = (2*4 + 2 * numThreads/32) * sizeof(float);

  auto displx = Matx13f(1, 0, 0) * trMat;
  auto disply = Matx13f(0, 1, 0) * trMat;

  preprocess_kernel<<< gridSize, blockSize, memBytes >>>(cudaPoints,
                                                          (float*)first.data,
                                                          (float*)window.data, window.step/sizeof(float),
                                                          row_stride, first.rows * row_stride,
                                                          make_float2(displx(0, 0), displx(0, 1)),
                                                          make_float2(disply(0, 0), disply(0, 1)),
                                                          count);



  // cout << first << endl;
  // exit(0);
  // printf("Here count!!! %d\n", count);
  // cudaDeviceSynchronize();

  // printf("CPU Effective value: %f\n", patch.ptr<float>(cudaPoints[0].y)[cudaPoints[0].x]);

  // for (int i = 200; i < 210; i++) {
  //   printf("CPU value %d: %f\n", i, ((float*)patch.data)[i]);
  // }
  // cout << patch << endl;

  // Scalar mean, std;
  // meanStdDev(first, mean, std);
  //
  // Mat m1;
  // first.copyTo(m1);
  // multiply(m1, m1, m1);
  //
  // printf("CPU Mean: %f Std: %f Sum: %f SQS: %f\n", (float)mean[0], (float)std[0],
  //                       (float)sum(first)[0],
  //                       (float)sum(m1)[0]);
}

template<class T>
__device__ void ternary_op(T &dest, T const& src1, T const& src2);

template <class T, decltype(&ternary_op<T>) f>
static __global__ void ternary_operation_kernel(T* dest, int dest_row_stride, int dest_stride,
                                const T* src1, int src1_row_stride, int src1_stride,
                                const T* src2, int src2_row_stride, int src2_stride,
                                int count) {

  const int IterationsPerThread = 4;
  const int RowThreadBlocksCount = blockDim.y;
  const int SubPatchCount = blockDim.z;

  const int TotalDestThreadStride = dest_row_stride * RowThreadBlocksCount * IterationsPerThread;

  const int subColIndex = threadIdx.x;
  const int rowIndex = threadIdx.y;
  const int subPatchIndex = threadIdx.z;
  const int patchBlockIndex = blockIdx.x;

  // Find current patch index
  int patchIdx =  subPatchIndex + SubPatchCount * patchBlockIndex;

  if (patchIdx >= count) // Exit if outside patches count
    return;


  T *row_dest = dest + (patchIdx * dest_stride);
  const T *row_src1 = src1 + (patchIdx * src1_stride);
  const T *row_src2 = src2 + (patchIdx * src2_stride);


  const int destThreadStride = dest_row_stride * RowThreadBlocksCount;
  const int src1ThreadStride = src1_row_stride * RowThreadBlocksCount;
  const int src2ThreadStride = src2_row_stride * RowThreadBlocksCount;


  int destOffset = subColIndex + (rowIndex * dest_row_stride);
  int src1Offset = subColIndex + (rowIndex * src1_row_stride);
  int src2Offset = subColIndex + (rowIndex * src2_row_stride);

  while (destOffset < TotalDestThreadStride) {
    f(row_dest[destOffset], row_src1[src1Offset], row_src2[src2Offset]);

    destOffset += destThreadStride;
    src1Offset += src1ThreadStride;
    src2Offset += src2ThreadStride;
  }
}

__forceinline__ __device__ void op_mul_complex(cuComplex &dest, cuComplex const& src1, cuComplex const& src2) {
  dest = cuCmulf(src1, cuConjf(src2));
}

__forceinline__ __device__ void op_div_complex(cuComplex &dest, cuComplex const& src1, cuComplex const& src2) {
  dest = (src2.x == 0 && src2.y == 0) ? make_float2(0, 0) : cuConjf(cuCdivf(src1, src2));
}

__forceinline__ __device__ void op_update_correlation(cuComplex &dest, cuComplex const& src1, cuComplex const& src2) {
  //G, A, h1, 0, true);
  const float rate = 0.125;
  const cuComplex h1 = cuCmulf(src1, cuConjf(src2));
  dest = cuCaddf(cuCmulf(dest, make_float2(1.0f-rate, 0)), cuCmulf(h1, make_float2(rate, 0)));
}

void cuda_funcs::convolve_cmplx(Mat const& src1, Mat const& src2, Mat &dest, int count, CONV_OPS op) {

  dim3 blockSize(32, 8, 4);
  dim3 gridSize((count + 3) / 4, 1, 1);

  int dest_stride = dest.step/sizeof(cuComplex);
  int src1_stride = src1.step/sizeof(cuComplex);
  int src2_stride = src2.step/sizeof(cuComplex);

  int dest_size = dest_stride * dest.rows;
  int src1_size = src1_stride * src1.rows;
  int src2_size = src2_stride * src2.rows;

  switch (op) {
    case OP_MUL: {
      ternary_operation_kernel<cuComplex, &op_mul_complex><<<gridSize, blockSize>>>((cuComplex*)dest.data, dest_stride, dest_size,
                                              (cuComplex*)src1.data, src1_stride, src1_size,
                                              (cuComplex*)src2.data, src2_stride, src2_size, count);
      break;
    }
    case OP_DIV: {
      ternary_operation_kernel<cuComplex, &op_div_complex><<<gridSize, blockSize>>>((cuComplex*)dest.data, dest_stride, dest_size,
                                              (cuComplex*)src1.data, src1_stride, src1_size,
                                              (cuComplex*)src2.data, src2_stride, src2_size, count);
      break;
    }
    case OP_UPD_CORR:
    case OP_UPD_CORR_G: {
      ternary_operation_kernel<cuComplex, &op_update_correlation><<<gridSize, blockSize>>>((cuComplex*)dest.data, dest_stride, dest_size,
                                              (cuComplex*)src1.data, src1_stride, (op == OP_UPD_CORR_G) ? 0 : src1_size,
                                              (cuComplex*)src2.data, src2_stride, src2_size, count);
      break;
    }
  }
  // cudaDeviceSynchronize();
}

static __global__ void max_std_kernel(PatchDesc *patch_desc,
                                  float const* mat,
                                  int row_stride, int stride,
                                  int count) {

  const int IterationsPerThread = 4;
  const int NumberOfColumns = blockDim.x;
  const int RowThreadBlocksCount = blockDim.y;
  const int SubPatchCount = blockDim.z;

  const int ThreadsPerPatch = RowThreadBlocksCount * NumberOfColumns;

  const int TotalPixelsPerPatch = ThreadsPerPatch * IterationsPerThread;

  const int ThreadPatchStride = row_stride * RowThreadBlocksCount;
  const int ThreadRowStride = RowThreadBlocksCount;

  const int WarpsPerPatch = ThreadsPerPatch / warpSize;

  const int subColIndex = threadIdx.x;
  const int rowIndex = threadIdx.y;
  const int subPatchIndex = threadIdx.z;
  const int patchBlockIndex = blockIdx.x;

  // single shared memory
  extern __shared__ float shared[];

  // Compute accumulators start address (skip mean and std)
  float * __restrict__ accum = shared + (SubPatchCount * 2);

  // Find current patch index
  int patchIdx =  subPatchIndex + patchBlockIndex * SubPatchCount;

  if (patchIdx >= count) // Exit if outside patches count
    return;

  // Find start pointer to the current patch
  const float * __restrict__ current_patch = mat + (patchIdx * stride);

  const int patch_thread_index = subColIndex + (rowIndex * NumberOfColumns);
  const int warpIndex = patch_thread_index/warpSize;

  // Compute current thread shared index => [0-256] first patch [256-512] second patch etc...
  const int patch_accum_index = (subPatchIndex * WarpsPerPatch) * 4;
  float * __restrict__ __mean_accum = accum + patch_accum_index + warpIndex;
  float * __restrict__ __sqmean_accum = accum + patch_accum_index + warpIndex + WarpsPerPatch;
  float * __restrict__ __max_val_accum = accum + patch_accum_index + warpIndex + WarpsPerPatch * 2;
  int * __restrict__ __max_idx_val = ((int*)accum) + patch_accum_index + warpIndex + WarpsPerPatch * 3;


  // Set accumulators to 0
  float mean_accum = 0;
  float sqmean_accum = 0;
  float max_val = -INFINITY;
  int max_idx[2] = {0, -1};


  int lOffset = 0;
  int rowOffset = rowIndex;

  int patchOffset = subColIndex + (rowIndex * row_stride);
  for (int regIdx = 0; regIdx < IterationsPerThread; regIdx++) {
    float pixel = current_patch[patchOffset];
    mean_accum += pixel;
    sqmean_accum += pixel*pixel;
    max_val = max(max_val, pixel);
    max_idx[pixel==max_val] = patchOffset;
    patchOffset += ThreadPatchStride;
  }
  __syncthreads();


  int laneidx = lane_id();
  warpReduceSumTwo(mean_accum, sqmean_accum);
  warpReduceMaxIdx(max_val, max_idx);
  if (!laneidx) {
    __mean_accum[0] = mean_accum;
    __sqmean_accum[0] = sqmean_accum;
    __max_val_accum[0] = max_val;
    __max_idx_val[0] = max_idx[1];
  }
  if (warpIndex) return;
  __syncthreads();


  // First warp
  int maxidx_accum[2];
  if (patch_thread_index < WarpsPerPatch) {
    float gmean_accum = accum[patch_accum_index + patch_thread_index];
    float gsqmean_accum = accum[patch_accum_index + patch_thread_index + WarpsPerPatch];
    float maxval_accum = accum[patch_accum_index + patch_thread_index + WarpsPerPatch * 2];
    maxidx_accum[1] = ((int*)accum)[patch_accum_index + patch_thread_index + WarpsPerPatch * 3];

    warpReduceSumTwo(gmean_accum, gsqmean_accum);
    warpReduceMaxIdx(maxval_accum, maxidx_accum);
    if (!patch_thread_index) {
      patch_desc[patchIdx] = { maxidx_accum[1], maxval_accum, gmean_accum, gsqmean_accum };
    }
  }

  maxidx_accum[1] = __shfl(maxidx_accum[1], 0); // Take maximum index from thread 0

  int rowi = patch_thread_index / 11 - 5;
  int coli = patch_thread_index % 11 - 5;
  float svalue = 0;
  float sqvalue = 0;
  for (int i = 0; i < 4; i++) {
    float value = current_patch[max(0, min(stride, maxidx_accum[1] + rowi * row_stride + coli))];
    svalue += value;
    sqvalue += value * value;
    rowi += 3;
  }
  warpReduceSumTwo(svalue, sqvalue);


  if (!patch_thread_index) {
    // printf("Value: %d Max(%d, %d) = %f\n", patchIdx, maxidx_accum[1]/row_stride, maxidx_accum[1]%row_stride, value);
    patch_desc[patchIdx].sum -= svalue;
    patch_desc[patchIdx].sqsum -= sqvalue;

    // printf("Maximum: %d %f %f\n", patchIdx, maxval_accum, current_patch[maxidx_accum[1]]);

    // // Array of 2 values, mean and std
    // float2 &meanStandardDeviation = ((float2*)shared)[subPatchIndex];
    //
    // float mean = meanStandardDeviation.x = mean_accum/TotalPixelsPerPatch;
    // // Compute std
    // float std = sqrt(sqmean_accum/TotalPixelsPerPatch - mean*mean);
    // const float eps = 1e-5;
    // float mult = 1.0f/(std+eps);
    // meanStandardDeviation.y = mult;
  }
  //
  // float2 meanStd = ((float2*)shared)[subPatchIndex];
  //
  // patchOffset = subColIndex + (rowIndex * row_stride);
  // for (int regIdx = 0; regIdx < IterationsPerThread; regIdx++) {
  //   // current_patch[patchOffset] = (logsRegs[regIdx] - meanStd.x) * meanStd.y * window[windowOffset];
  //   patchOffset += ThreadPatchStride;
  // }
}

PatchDesc *maxs;

const PatchDesc* cuda_funcs::max_std_process(Mat &first_patch, int count) {

  if (!maxs) {
    cudaMallocManaged((void**)&maxs, count * sizeof(PatchDesc));
  }

  dim3 blockSize(32, 8, 4);
  int numThreads = 1024;//blockSize.x * blockSize.y * blockSize.z;
  dim3 gridSize((count + 3) / 4, 1, 1);

  int row_stride = first_patch.step/sizeof(float);
  int memBytes = (2*4 + 4 * numThreads/32) * sizeof(float);
  max_std_kernel<<< gridSize, blockSize, memBytes >>>(maxs,
                                                          (float*)first_patch.data,
                                                          row_stride, first_patch.rows * row_stride,
                                                          count);
  cudaDeviceSynchronize();

  return maxs;
}

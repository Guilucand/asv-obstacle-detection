#include <opencv2/opencv.hpp>
#include <fftw3.h>

#ifdef CUDA
#include <cufft.h>
#include <cuda_runtime.h>

#ifdef JETSON
#include <opencv2/gpu/gpu.hpp>
#else
#include <opencv2/core/cuda.hpp>
#endif

#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

using namespace cv;

#ifdef JETSON
namespace cuda = cv::gpu;
#endif

#include "fft.h"


bool init = false;

FFT::FFT(FFT_MODE mode, FFT_DIRECTION dir, Mat source, Mat dest, int count) {

  // std::cout << "Dest cols before: " << dest.cols << std::endl;
  // std::cout << "Src cols before: " << source.cols << std::endl;
  // if (dir_ == DIR_FORWARD)
  //   dest.cols = (dest.cols-1) * 2;
  // else
  //   source.cols = (source.cols-1) * 2;

  // std::cout << "Dest cols: " << dest.cols << std::endl;
  // std::cout << "Src cols: " << source.cols << std::endl;


  mode_ = mode;
  dir_ = dir;
  source_ = source;
  dest_ = dest;
  count_ = count;


  int in_batch_stride = (int)source_.rows * (int)(source_.step/sizeof(float));
  int out_batch_stride = (int)dest_.rows * (int)(dest_.step/sizeof(float));

  int size[2] = {source_.rows, source_.cols};
  int instrides[2] = {in_batch_stride, (int)(source_.step/sizeof(float))};
  int outstrides[2] = {out_batch_stride, (int)(dest_.step/sizeof(float))};

  switch (dir_) {
    case DIR_FORWARD:
      out_batch_stride /= 2;
      outstrides[0] /= 2;
      outstrides[1] /= 2;
    break;
    case DIR_BACKWARD:
      in_batch_stride /= 2;
      instrides[0] /= 2;
      instrides[1] /= 2;
      size[1] *= 2;
    break;
 }

  switch (mode) {

#ifdef CUDA
    case FFT_CUFFT: {
      int res0 = cufftPlanMany(&cudaHandle, 2, size, instrides, 1, in_batch_stride, outstrides, 1, out_batch_stride, (dir == DIR_FORWARD) ? CUFFT_R2C : CUFFT_C2R, count);
      printf("Result: %d and %d %d %d %d %d %d\n", res0, CUFFT_SUCCESS, CUFFT_ALLOC_FAILED, CUFFT_INVALID_VALUE, CUFFT_INTERNAL_ERROR, CUFFT_SETUP_FAILED, CUFFT_INVALID_SIZE);
      if (res0 != CUFFT_SUCCESS)
        exit(0);
    }
    break;
#endif
    case FFT_FFTW: {
      normalized = false;
      if (!init) {
        init = true;
        fftwf_init_threads();
        fftwf_plan_with_nthreads(8);
      }

      switch (dir) {
        case DIR_FORWARD:
          fftwPlan = fftwf_plan_many_dft_r2c(2, size, count, (float*)source.data, instrides, 1, in_batch_stride, (float(*)[2])dest.data, outstrides, 1, out_batch_stride, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        break;
        case DIR_BACKWARD:
          fftwPlan = fftwf_plan_many_dft_c2r(2, size, count, (float(*)[2])source.data, instrides, 1, in_batch_stride, (float*)dest.data, outstrides, 1, out_batch_stride, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        break;
      }
    }
    break;

    case FFT_OPENCV:
    case FFT_OPENCV_PARALLEL:
    case FFT_OPENCV_CUDA:
      normalized = false;
      break;
  }
}

void FFT::execute() {

  // Mat dest123 = dest_;
  // dest123.data += (dest.rows * dest.step/sizeof(float)) * 311;

  switch (mode_) {
    case FFT_FFTW: {
        Mat dest1;
        dft(source_, dest1, DFT_COMPLEX_OUTPUT);

        fftwf_execute(fftwPlan);

        // if (dir_ == DIR_FORWARD) {
        //   std::cout << "Sum fftw = " << dest_({0, 0, 16, 32}) << std::endl << std::endl;
        //   std::cout << "Sum opcv = " << dest1({0, 0, 16, 32}) << std::endl << std::endl << std::endl << std::endl;
        //
        //   Mat a, b;
        //   a = dest1({0, 0, 16, 32});
        //   b = dest_({0, 0, 16, 32});
        //
        //   std::cout << "Sizes: " << dest_.size() << " and " << dest1.size() << std::endl;
        //   std::cout << "Types: " << dest_.type() << " and " << dest1.type() << std::endl;
        //
        //   Mat xx = abs(a-b);
        //   double mx;
        //   minMaxLoc(xx, NULL, &mx, NULL, NULL);
        //
        //   std::cout << "Diff = " << mx << std::endl << std::endl << std::endl << std::endl;
        //
        //   std::cout << "Sums :: " << sum(dest_({0, 0, 16, 32}))[0] << " and " << sum(dest1({0, 0, 16, 32}))[0] << std::endl;
        // }
      }
      break;
    case FFT_OPENCV:
    case FFT_OPENCV_PARALLEL: {
      for (int i = 0; i < count_; i++) {
        Mat source = source_;
        Mat dest = dest_;
        source.data += (source.rows * source.step) * i;
        dest.data += (dest.rows * dest.step) * i;

        switch (dir_) {
          case DIR_FORWARD: {
            void * data = dest.data;
            dft(source, dest, DFT_COMPLEX_OUTPUT);
            if (i == 0) {
              // printf("Pointers: %p %p Count:%d\n", data, dest.data, count_);
              // printf("Pointers: %p %p Count:%d\n", source.data, source.data, count_);
            }
            break;
          }
          case DIR_BACKWARD:
            idft(source, dest, DFT_REAL_OUTPUT);
          break;
        }
      }
    }
    break;

#ifdef CUDA
    case FFT_OPENCV_CUDA: {
      cuda::GpuMat source_gpu_ (source_.rows, source_.cols, source_.type(), source_.data, source_.step);
      cuda::GpuMat dest_gpu_ (dest_.rows, dest_.cols, dest_.type(), dest_.data, dest_.step);

      // void *a, *b;
      // cudaMallocManaged(&a, 1000000);
      // cudaMallocManaged(&b, 1000000);
      //
      // std::cout << "Copying: " << cudaMemcpy2D(dest_.data, dest_gpu_.step, b+1, source_gpu_.step, source_gpu_.cols * source_gpu_.elemSize(), source_gpu_.rows, cudaMemcpyDeviceToDevice)
      //     << " " << (void*)dest_gpu_.data << std::endl;

      for (int i = 0; i < count_; i++) {
        cuda::GpuMat source = source_gpu_;
        cuda::GpuMat dest = dest_gpu_;
        source.data += (source.rows * source.step) * i;
        dest.data += (dest.rows * dest.step) * i;

        switch (dir_) {
          case DIR_FORWARD: {
            void * data = dest.data;
            try {
              cuda::dft(source, dest, source.size());

              // Mat source__ (source.rows, source.cols, source.type(), source.data, source.step);
              // Mat dest__ (dest.rows, dest.cols, dest.type(), dest.data, dest.step);
              //
              // dft(source__, dest__, DFT_COMPLEX_OUTPUT);
            }
            catch (Exception) {
              fprintf(stderr,"GPUassert: %s \n", cudaGetErrorString(cudaGetLastError()));
              exit(1);
            }
            if (i == 0) {
              // printf("Pointers: %p %p Count:%d\n", data, dest.data, count_);
              // printf("Pointers: %p %p Count:%d\n", source.data, source.data, count_);
            }
            break;
          }
          case DIR_BACKWARD:
            Mat source__ (source.rows, source.cols, source.type(), source.data, source.step);
            Mat dest__ (dest.rows, dest.cols, dest.type(), dest.data, dest.step);

            idft(source__, dest__, DFT_REAL_OUTPUT);
          break;
        }
      }
      cudaDeviceSynchronize();
    }
    break;


    case FFT_CUFFT: {
      int res;
      switch (dir_) {
        case DIR_FORWARD:
          cufftExecR2C(cudaHandle, (cufftReal*)source_.data, (cufftComplex*)dest_.data);
        break;
        case DIR_BACKWARD:
          cufftExecC2R(cudaHandle, (cufftComplex*)source_.data, (cufftReal*)dest_.data);
        break;
      }
      // cudaDeviceSynchronize();
      if (res != CUFFT_SUCCESS) {
        printf("Result: %d\n", res);
        exit(0);
      }

    }
    break;
#endif
    // {
    //   switch (dir_) {
    //     case DIR_FORWARD: {
    //       void * data = dest_.data;
    //       // dft(source_, dest_, DFT_COMPLEX_OUTPUT);
    //       cuda::dft(source_, dest_, source_.size());
    //       printf("Pointers: %p %p\n", data, dest_.data);
    //       break;
    //     }
    //     case DIR_BACKWARD:
    //       // idft(source_, dest_, DFT_REAL_OUTPUT);
    //     break;
    //   }
    // }
    // break;
  }

  // std::cout << "Sum = " << sum(dest_)[0] << std::endl;
}

#ifdef CUDA
//
static Mat B[1024];
// void DFT(Mat const& A, int count) {
//   for (int i = 0; i < count; i++) {
//     dft(A, B[i%1024], DFT_COMPLEX_OUTPUT);
//   }
//   int row = 10;
//   int column = 10;
//   printf("Value0: %f\n", ((float*)&(B[0].data[row * B[0].step]))[column]);
// }

void PDFT(Mat const& A, int count) {
  #pragma omp parallel for
  for (int i = 0; i < count; i++) {
    dft(A, B[i%1024], DFT_COMPLEX_OUTPUT);
  }
  int row = 10;
  int column = 10;
  printf("Value0: %f\n", ((float*)&(B[0].data[row * B[0].step]))[column]);
}

float mat[2048 * 2048];
float mat2[2048 * 2048];

float *gpuIMGS;
float *gpuFFT;
cufftHandle pF;

int step_old;

void GDFT(Mat const& A, int count) {
  cuda::GpuMat gA, gB;
  gA.upload(A);

  if (!gpuFFT) {
    cudaMalloc(&gpuFFT, gA.rows * gA.step * 2 * count * sizeof(float) + 1024);
    cudaMalloc(&gpuIMGS, gA.rows * gA.step * count * sizeof(float) + 1024);
  }

  if (!pF) {
    int batch_stride = (int)gA.rows * (int)(gA.step/sizeof(float));

    int size[2] = {gA.rows, gA.cols};
    int instrides[2] = {batch_stride, (int)(gA.step/sizeof(float))};
    int outstrides[2] = {batch_stride*2, (int)(gA.step/sizeof(float))};

    int res0 = cufftPlanMany(&pF, 2, size, instrides, 1, batch_stride, outstrides, 1, batch_stride*2, CUFFT_R2C, count);
    printf("Result: %d and %d %d %d %d %d %d\n", res0, CUFFT_SUCCESS, CUFFT_ALLOC_FAILED, CUFFT_INVALID_VALUE, CUFFT_INTERNAL_ERROR, CUFFT_SETUP_FAILED, CUFFT_INVALID_SIZE);
    if (res0 != CUFFT_SUCCESS)
      exit(0);

  }
  if (step_old && gA.step != step_old) {
    printf("Size mismatch: Steps: %lu and %d\n", gA.step, step_old);
    exit(0);
  }
  step_old = gA.step;

  cudaMemcpy(gpuIMGS, gA.data, gA.rows * gA.step, cudaMemcpyDeviceToDevice);
  int res1 = cufftExecR2C(pF, (cufftReal*)gpuIMGS, (cufftComplex*)gpuFFT);
  if (res1 != CUFFT_SUCCESS) {
    printf("Result: %d\n", res1);
    exit(0);
  }
  int row = 10;
  int column = 10;
  cudaMemcpy(mat2, gpuFFT, gA.rows * gA.step * 2, cudaMemcpyDeviceToHost);
  printf("Value3: %f at address %p\n", ((float*)&(((char*)mat2)[row * gA.step * 2]))[column], &(((float*)&(((char*)mat2)[row * gA.step * 2]))[column]));

}

void GCVDFT(Mat const& A, int count) {
  cuda::GpuMat gA, gB;
  gA.upload(A);
  for (int i = 0; i < count; i++) {
    cuda::dft(gA, gB, gA.size());
  }
  Mat B;
  gB.download(B);
  int row = 10;
  int column = 10;
  printf("Value2: %f\n", ((float*)&(B.data[row * B.step]))[column]);
}

fftwf_plan wplan;
float *wFFT, *wIMGS;

void WDFT(Mat const& A, int count) {
  if (!wFFT) {
    wFFT = (float*)malloc(A.rows * A.step * 2 * count * sizeof(float) + 1024);
    wIMGS = (float*)malloc(A.rows * A.step * count * sizeof(float) + 1024);
  }

  if (!wplan) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(8);
    int batch_stride = (int)A.rows * (int)(A.step/sizeof(float));

    int size[2] = {A.rows, A.cols};
    int instrides[2] = {batch_stride, (int)(A.step/sizeof(float))};
    int outstrides[2] = {batch_stride*2, (int)(A.step/sizeof(float))};

    wplan = fftwf_plan_many_dft_r2c(2, size, count, wIMGS, instrides, 1, batch_stride, (float(*)[2])wFFT, outstrides, 1, batch_stride*2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  }
  memcpy(wIMGS, A.data, A.rows * A.step);
  fftwf_execute(wplan);
  printf("Step: %d\n", (int)A.step);


  int row = 10;
  int column = 10;
  printf("Value1: %f\n", ((float*)&(((char*)wFFT)[row * A.step * 2]))[column]);
}

#endif

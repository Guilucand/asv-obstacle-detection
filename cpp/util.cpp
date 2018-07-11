#include <opencv2/opencv.hpp>

#include "util.h"
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

#ifdef CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

using namespace cv;
using namespace std;

float* allocate(int rows, int cols, int step, int count) {
  // printf("Step: %d\n", step);
  int misalign = 0;
  int size = count * rows * step * sizeof(float) + misalign;

#ifdef CUDA
  float *ptr;
  cudaMallocManaged((void**)&ptr, size);
  cudaMemset(ptr, 0, size);
#else
  float *ptr = (float*)((char*)memalign(1024, size) + misalign);
  memset(ptr, 0, size);
#endif

  return ptr;
}

int align(int size, int alignment) {
  return size + (alignment - size%alignment) % alignment;
}


void draw_text(Mat image, string text, Point origin) {
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1.0;
  int thickness = 2;
  putText(image, text, origin, fontFace, fontScale,
          Scalar::all(255), thickness+1, 8);
  putText(image, text, origin, fontFace, fontScale,
          Scalar::all(64), thickness, 8);
}


void Canny(cv::Mat image) {
  Mat workImg;
  // Step 1: Noise reduction
  cv::GaussianBlur(image, workImg, cv::Size(0, 0), 1.0);
  auto size  = 3;


  // Step 2: Calculating gradient magnitudes and directions
  Mat magX = Mat(image.rows, image.cols, CV_32F);
  Mat magY = Mat(image.rows, image.cols, CV_32F);
  cv::Sobel(workImg, magX, CV_32F, 1, 0, size);
  cv::Sobel(workImg, magY, CV_32F, 0, 1, size);

  Mat direction = Mat(workImg.rows, workImg.cols, CV_32F);
  cv::divide(magY, magX, direction);

  Mat sum = Mat(workImg.rows, workImg.cols, CV_32F);
  Mat prodX = Mat(workImg.rows, workImg.cols, CV_32F);
  Mat prodY = Mat(workImg.rows, workImg.cols, CV_32F);
  cv::multiply(magX, magX, prodX);
  cv::multiply(magY, magY, prodY);
  sum = prodX + prodY;
  cv::sqrt(sum, sum);

  // sum.copyTo(image);
  // cv::imshow("Canny_raw", sum/256.0);

  #pragma omp parallel for
  for (int i = 0; i < image.rows; i++) {
    float *prow = sum.ptr<float>(max(0, i-1));
    float *row = sum.ptr<float>(i);
    float *nrow = sum.ptr<float>(min(i+1, image.rows-1));
    float *dir = direction.ptr<float>(i);
    for (int j = 0; j < image.cols; j++) {
      bool nlast = j != image.cols-1;
      if (dir[j] > 2.4142 || dir[j] < -2.4142) { // 67.5 < a || a < -67.5
        if (row[j] < prow[j] || row[j] < nrow[j])
          row[j] = 0;
      }
      else if (dir[j] > 0.4142) {
        if ((nlast && row[j] < prow[j + 1]) || (j && row[j] < nrow[j - 1]))
          row[j] = 0;
      }
      else if (dir[j] < 0.4142) {
        if ((nlast && row[j] < row[j + 1]) || (j && row[j] < row[j - 1]))
          row[j] = 0;
      }
      else {
        if ((j && row[j] < prow[j - 1]) || (nlast && row[j] < nrow[j + 1]))
          row[j] = 0;
      }
    }
  }
  sum.copyTo(image);
  // imshow("sobely", magY);
}

void blend(cv::Mat dst, cv::Mat src) {

  #pragma omp parallel for
  for (int i = 0; i < dst.rows; i++) {
    Vec4f *srcr = src.ptr<Vec4f>(i);
    Vec3b *dstr = dst.ptr<Vec3b>(i);
    for (int j = 0; j < dst.cols; j++) {
      float alpha = srcr[j][3] / 255.0;
      dstr[j] = dstr[j] * (1.0f-alpha) + Vec3b(srcr[j][0], srcr[j][1], srcr[j][2]) * alpha;
    }
  }
}

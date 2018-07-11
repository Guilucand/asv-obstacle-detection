#include <opencv2/opencv.hpp>
#include <algorithm>

#ifdef CUDA
//#include <cuda_runtime.h>
#include <cufft.h>
#endif

#include <fftw3.h>

using namespace cv;
using namespace std;

// #include "util.h"
#include "mosse.h"


Mat gray;

Mat to_gray(Mat const& frame) {

  gray.create(frame.rows, frame.cols, CV_32FC1);

  for (int i = 0; i < frame.rows; i++) {
    const uint8_t (*row)[3] = frame.ptr<uint8_t[3]>(i);
    float *out_row = gray.ptr<float>(i);
    for (int j = 0; j < frame.cols; j++) {
      out_row[j] = row[j][0] * (0.114f/256.0f) + row[j][1] * (0.587f/256.0f) + row[j][2] * (0.299f/256.0f);
    }
  }
  return gray;
}


// int step1, wid, hei, kernel_size;
int kernel_size;
FFT *fft;
FFT *fft_inv;
FFT *fft_horiz, *fft_horiz_inv;
Mat filter;

Mat window;
Mat G;

void genGaussian(Mat &mat, float* gauss, int ksize) {
  for (int i = 0; i < ksize; i++) {
    float *row = (float*)&(mat.data[i * mat.step]);
    for (int j = 0; j < ksize; j++) {
      row[j] = gauss[i] * gauss[j];
    }
  }
}

Mat getGaussianFilterDFT(float sigma, int width, int height) {

  int kernel_size_old = (((sigma-0.8f)/0.3f+1.0f)*2.0f + 0.5f);
  kernel_size = cvRound(sigma*6 + 1)|1;
  kernel_size |= 1;

  printf("Kernel size: %d old: %d\n", kernel_size, kernel_size_old);

  Size zero(0, 0);
  width += kernel_size;
  height += kernel_size;
  width = getOptimalDFTSize(width);
  height = getOptimalDFTSize(height);
  int step = width;

  printf("Width: %d Height: %d\n", width, height);

  // memcpy(filter, kernel.data, kernel.rows * sizeof(float)); // DEBUG: Why rows and not cols?????

  Mat kernel = getGaussianKernel(kernel_size, sigma, CV_32F);

  int alignment = 32;

  if (!fft) {
    step = align(step, alignment);
    int step_bytes = step * sizeof(float);
    Mat src(height, width, CV_32FC1, allocate(height, width, step), step_bytes);
    Mat dst(height, width, CV_32FC2, allocate(height, width, step * 2), step_bytes * 2);
    // for (int i = 1; i < 1024; i *= 2) {
    //   printf("Src divisible by %d: %llu\n", i, ((unsigned long long int)src.data) % i);
    //   printf("Dst divisible by %d: %llu\n", i, ((unsigned long long int)dst.data) % i);
    //   printf("Step divisible by %d: %llu\n", i, ((unsigned long long int)step_bytes) % i);
    // }
    // exit(0);

    fft = new FFT(FFT_FFTW, DIR_FORWARD, src, dst);
    fft_inv = new FFT(FFT_FFTW, DIR_BACKWARD, dst, src);
  }
  genGaussian(fft->source_, (float*)kernel.data, kernel_size);
  fft->execute();
  filter = fft->dest_.clone();

  return filter;
}

Mat pad_frame(Mat const& frame) {
  // cout << "Scale: " << frame.size() << " and " << fft->source_.size() << endl;
  memset(fft->source_.data, 0, fft->source_.step * fft->source_.rows);
  for (int i = 0; i < frame.rows; i++) {
    const float *row = frame.ptr<float>(i);
    float *out_row = fft->source_.ptr<float>(i);
    memcpy(out_row, row, frame.cols * sizeof(float));
  }

  return fft->source_;
}

Mat unpad_frame(Mat &frame, Mat &fframe, int divisor = 2) {

  int leftx = 0;
  int lefty = 0;

  if (divisor == 2) {
    leftx += frame.cols/4;
    lefty += frame.rows/4;
  }
  // randu(frame, -0.2, 0.2);
  // randu(fframe, -0.2, 0.2);
  for (int i = 0; i < frame.rows/divisor; i++) {
    const float *row = fft_inv->dest_.ptr<float>(i + kernel_size/2);
    float *out_row = frame.ptr<float>(i + lefty) + leftx;
    float *out_row2 = fframe.ptr<float>(frame.rows/divisor - i - 1 + lefty) + leftx;
    memcpy(out_row, row + kernel_size/2, frame.cols/divisor * sizeof(float));
    memcpy(out_row2, row + kernel_size/2, frame.cols/divisor * sizeof(float));
  }
  return frame;
}


void fct_fftshift(cv::Mat& src)
{
    int cx = src.cols/2;
    int cy = src.rows/2;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size(), INTER_LINEAR, BORDER_WRAP);
    return trans_mat;
}

Point startHorizon,
      endHorizon;



bool initk;
Mat kernel, blurred, flipped, B, F;
Mat draw_horizon(Mat const& frame, Mat const& outframe, float sigma) {
  Mat grayscale;
  // cvtColor(frame, grayscale, COLOR_BGR2GRAY);
  // grayscale.convertTo(grayscale, CV_32FC1);
  // multiply(grayscale, 1.0f/256.0f, grayscale);
  // imshow("GSFLOAT", grayscale);
  // float *out_row = gray + (grayscale.step/sizeof(float)) * grayscale.rows/2;
  // out_row[grayscale.cols / 2] = 256.0f;

  int fftscalar = 1;

  if (!initk) {
    Size winSize(frame.cols * fftscalar, frame.rows * fftscalar);
    initk = true;
    kernel = getGaussianFilterDFT(sigma, frame.cols, frame.rows);
    createHanningWindow(window, winSize, CV_32FC1);
    G = Mosse::gen_corr_target(winSize.width, winSize.height, 2.0);
  }
  grayscale = to_gray(frame);
  // if (startHorizon.y) {
  //     line(grayscale, startHorizon, endHorizon, 1, 1);
  // }
  // Mosse::preprocess(grayscale, window);
  Mat pframe = pad_frame(grayscale);
  fft->execute();

  Mat filter_mat = fft->dest_;

  // printf("%d %d\n", kernel.size().width, filter_mat.size().width);

  mulSpectrums(filter_mat, kernel, filter_mat, 0, false);
  filter_mat /= filter_mat.rows * filter_mat.cols;
  // filter_mat = kernel;
  fft_inv->execute();
  Mat sol = fft_inv->dest_;

  if (!fft_horiz) {
    int rows = grayscale.rows * fftscalar;
    int cols = grayscale.cols * fftscalar;
    int step = align(cols, 32) * sizeof(float);
    int batchsz = rows * step / sizeof(float);
    float* ptr = allocate(rows, cols, step, 2);
    float* out_ptr = allocate(rows, cols, step*2, 2);

    blurred = Mat(rows, cols, CV_32FC1, ptr, step);
    flipped = Mat(rows, cols, CV_32FC1, ptr + batchsz, step);

    B = Mat(rows, cols, CV_32FC2, out_ptr, step * 2);
    F = Mat(rows, cols, CV_32FC2, out_ptr + batchsz * 2, step * 2);
    fft_horiz = new FFT(FFT_FFTW, DIR_FORWARD, blurred, B, 2);
    fft_horiz_inv = new FFT(FFT_FFTW, DIR_BACKWARD, B, blurred, 1);
  }
  blurred = Mat::zeros(blurred.rows, blurred.cols, CV_32F);
  flipped = Mat::zeros(blurred.rows, blurred.cols, CV_32F);
  unpad_frame(blurred, flipped, fftscalar);


  // blurred.copyTo(flipped);
  // flip(blurred, flipped, 0);
  // fct_fftshift(flipped);

  // GaussianBlur(grayscale, blurred, zero, 12.0);
  // sepFilter2D(grayscale, blurred, -1, filter, filter);
  // GaussianBlur(grayscale, blurred2, ksize, 12.0);
  // imshow("blurred", blurred);
  // Mat back = blurred.clone();
  // imshow("grayscale2", blurred2);

  // flip(blurred, flipped, 0);


  Size size = blurred.size();
  // copyMakeBorder(blurred, blurred, 0, size.height, 0, size.width, BORDER_CONSTANT);
  // copyMakeBorder(flipped, flipped, 0, size.height, 0, size.width, BORDER_CONSTANT);

  size = blurred.size();
  float energy = norm(blurred);
  energy *= energy;
  fft_horiz->execute();


  // dft(blurred, B, DFT_COMPLEX_OUTPUT);
  // idft(B, blurred, DFT_SCALE | DFT_REAL_OUTPUT);
  // idft(F, flipped, DFT_SCALE | DFT_REAL_OUTPUT);
  // imshow("blurred", blurred);
  // imshow("flipped", flipped);
  // imshow("flipped", flipped);

  Mat bOrig;
  B.copyTo(bOrig);

  // cout << B.size() << " and " << G.size() << endl;
  Mat h1, h2;
  // mulSpectrums(G, B, h1, 0, true);
  // mulSpectrums(B, B, h2, 0, true);
  // Mosse::divSpec(h1, h2, B);

  mulSpectrums(B, F, B, 0, true);
  // mulSpectrums(B, bOrig, B, 0, true);
  B /= B.rows * B.cols;

  fft_horiz_inv->execute();
  multiply(blurred, window, blurred);
  // idft(B, blurred, DFT_SCALE | DFT_REAL_OUTPUT);
  // imshow("blurred", blurred);

  Mat reduced;

  reduce(blurred, reduced, 1, CV_REDUCE_AVG);

  double minVal, maxVal;
  Point minPos, maxPos;
  minMaxLoc(reduced, &minVal, &maxVal, &minPos, &maxPos);

  if (maxPos.x > size.width/2)
    maxPos.x -= size.width/2;
  if (maxPos.y > size.height/2)
    maxPos.y -= size.height/2;

  maxPos.y += size.height/2;

  blurred -= minVal;
  blurred /= (maxVal-minVal);
  // printf("Max val: %f\n", maxVal);
  // fct_fftshift(blurred);
  // imshow("Blurred", blurred);

  // printf("Maxloc: %d %d\n", maxPos.x, maxPos.y);

  // print(mval/energy, mx, my)
  // # print(blurred.shape)
  // # flipped = np.roll(flipped, -mx//2, 1)
  // translateImg(flipped, maxPos.x, maxPos.y);
  // # cv2.imshow('flipped', flipped)#cv2.cvtColor(new_startframe, cv2.COLOR_BGR2GRAY)))#
  //
  // imshow("flipped", flipped);
  // imshow("total", back * 0.8f + flipped * 0.2f);

  int horizon = size.height/2/fftscalar + maxPos.y/2 - size.height/2;
  startHorizon = Point(0, horizon);
  endHorizon = Point(size.width, horizon);
  Scalar color(0, 255, 255);
  line(const_cast<Mat&>(outframe),startHorizon, endHorizon, color, 5);
  return sol;

}

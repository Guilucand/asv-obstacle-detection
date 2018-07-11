#pragma once
#include <opencv2/opencv.hpp>
#include "mosse.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


namespace cuda_funcs {
  struct PatchDesc {
    int coords;
    float maxValue;
    float sum;
    float sqsum;
  };

  void preprocess(std::vector<MosseWindow*> const& windows, cv::Size winSize, cv::Matx33f trMat, cv::Mat const& patch, cv::Mat &first, cv::Mat &window);


  enum CONV_OPS {
    OP_MUL,
    OP_DIV,
    OP_UPD_CORR,
    OP_UPD_CORR_G,
  };

  void convolve_cmplx(cv::Mat const& src1, cv::Mat const& src2, cv::Mat &dest, int count, CONV_OPS op = OP_MUL);

  const PatchDesc* max_std_process(cv::Mat &first_patch, int count);
}

#pragma once
#include <vector>
#include <map>
#include <stdio.h>
#include "fft.h"
#include "util.h"
#include "image_stab.h"
#include "smooth.h"

enum MOSSE_INIT_STATUS {
  MINIT_UNINITIALIZED,
  MINIT_RESETTED,
  MINIT_INITIALIZED
};

struct MosseWindow {
  cv::Point position;
  cv::Point begin_position;
  cv::Point displacement;
  cv::Point delta;
  cv::Mat H;
  cv::Mat H1;
  cv::Mat H2;
  int training_left;
  int reInit = 0;
  cv::Mat dbgPatch;
  bool frame_good;
  bool perm_deleted;
  bool now_outside;

  int outside_time;
  int notgood_time;
  int good_time;
  int mean_goodtime;
  int alive_time;
  int index;

  MOSSE_INIT_STATUS init_status;
  // MosseWindow* neighb[4] = {0};

  inline bool should_evaluate() {
    return !now_outside && !perm_deleted && (init_status != MINIT_UNINITIALIZED);
  }
};

class HorizonLine {
public:
  void add(std::vector<cv::Point> const& horizline, cv::Matx33f local);
  void draw(cv::Mat &frame, int startidx);
  void setg(cv::Matx33f global);

private:
  std::map<int, std::queue<int> > points;
  cv::Matx33f global;
};

template<class T>
T clamp(T d, T min, T max) {
  const T t = d < min ? min : d;
  return t > max ? max : t;
}

template <class T = float>
void extractSubMatTransform(cv::Mat const& source, cv::Size winSize, cv::Point center, cv::Mat &dest, cv::Matx33f transform) {
  cv::Point start = center-cv::Point(winSize.width/2, winSize.height/2);

  auto tstart = (cv::Matx13f(start.x, start.y, 1) * transform).get_minor<1, 2>(0, 0);
  auto tnextx = (cv::Matx13f(1, 0, 0) * transform).get_minor<1, 2>(0, 0);
  auto tnexty = (cv::Matx13f(0, 1, 0) * transform).get_minor<1, 2>(0, 0);

  cv::Matx12f psize = cv::Matx12f(source.cols-1, source.rows-1);

  for (int i = 0; i < winSize.height; i++) {
    T *row = dest.ptr<T>(i);
    auto startr = tstart + tnexty * i;
    for (int j = 0; j < winSize.width; j++) {
      // printf("Patch (%d, %d) => (%f, %f)\n", j, i, startr(0, 0), startr(0, 1));
      row[j] = source.at<T>(clamp(startr(0, 1), 0.0f, psize(0, 1)), clamp(startr(0, 0), 0.0f, psize(0, 0)));
      startr += tnextx;
    }
  }
}

class Mosse {
private:
  cv::Mat window;
  cv::Mat G; // FFT of gaussian expected filter
  cv::Size winSize;
  cv::Size frameSize;
  ::Allocator<MosseWindow> allocator;
  FFT *fft;
  FFT *fft_back;
  cv::Mat *fft_in;
  cv::Mat *fft_out;
  cv::Mat *winH;
  cv::Mat *h1Tmp;
  cv::Mat *h2Tmp;
  cv::Matx33f trMat;
  TransformMat rot_matrix;

public:
  std::vector<MosseWindow*> mosseWindows;


  int since_reset;
  static void preprocess(cv::Mat &image, cv::Mat const& window);
  static cv::Mat gen_corr_target(int width, int height, float sigma);
  static void divSpec(cv::Mat const& A, cv::Mat const& B, cv::Mat &C);
  cv::Size getWindowsCount();

  Mosse(int width, int height, int frame_width, int frame_height);
  void updateKernel(MosseWindow &win);

  inline void initMosse(MosseWindow &win);

  int updateWindow(cv::Mat &frame, MosseWindow &win, float psr_th, int state, float rate = 0.125);
  double convolve_step(MosseWindow &win);
  double correlate(cv::Mat &patch, MosseWindow &win);
  void refresh(cv::Mat mask, TransformMat trmat);
  void reset(cv::Mat mask, TransformMat trmat);
  cv::Mat update(cv::Mat frame, cv::Mat outframe, float psr_th, TransformMat trmat);

  void addWindow(cv::Point coords);

  void displaydbg(cv::Point coords);

  void reset_history();

  std::vector<std::vector<cv::Point> > horizhistory;

  SmoothLine smoothLine;
  int dbgidx;
  int max_windows;
};

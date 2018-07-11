#include <opencv2/opencv.hpp>
#include <omp.h>

#ifdef CUDA

#ifdef JETSON
#include <opencv2/gpu/gpu.hpp>
#else
#include <opencv2/core/cuda.hpp>
#endif

#include "cuda_funcs.h"

#endif

#include <stdio.h>
#include <complex.h>
#include "estimate.h"
#include "mosse.h"

using namespace cv;
using namespace std;

// #include <easy/profiler.h>

#define MAX_DELTIME 1
#define TRAINING_TIME 7
#define OVERLAP_DIVISOR 2


#define BEGIN_YIELD(x) switch(x) { case MINIT: {
#define END_YIELD() } }

enum MOSSE_STAGES {
  MINIT = 0,
  MSTAGE1_FFT,
  MSTAGE2_FFT_INV,
  MSTAGE3_FFT,
};


#define DO_FFT(stage) \
    return 1; \
    } case (stage): {

#define DO_FFT_INV(stage) \
    return 2; \
    } case (stage): {

void extractSubMat(Mat const& source, Size winSize, Point center, Mat &dest) {
  Point start = center-Point(winSize.width/2, winSize.height/2);
  Mat tmp = Mat(winSize.height, winSize.width, CV_32FC1, const_cast<float*>(source.ptr<float>(start.y)+start.x), source.step);
  tmp.copyTo(dest);
  // getRectSubPix(source, winSize, center, dest);
}

void Mosse::preprocess(Mat &image, Mat const& window) { // TODO: optimize with cuda kernels
  for(int i = 0; i < image.rows; i++){
    float *row = image.ptr<float>(i);
    for(int j = 0;j < image.cols; j++){
        row[j] = log(row[j]+1.0f);//conj(rowa[j] / rowb[j]);
    }
  }
  // medianBlur(image, image, 3);
  // add(image, 1.0f, image);
  // log(image, image);
  Scalar mean, std;
  meanStdDev(image, mean, std);
  // printf("Stddev: %f\n", std[0]);


  float eps = 1e-5;
  float mult = 1.0f/(std[0]+eps);
  for(int i = 0; i < image.rows; i++){
    float *row = image.ptr<float>(i);
    const float *win = window.ptr<float>(i);
    for(int j = 0;j < image.cols; j++){
        row[j] = (row[j]-mean[0]) * win[j] * mult; //log(row[j]+1.0f);//conj(rowa[j] / rowb[j]);
    }
  }
  // add(image, -mean[0], image);
  // multiply(image, 1.0f/(std[0]+eps), image);
  // multiply(image, window, image);
}


Mat Mosse::gen_corr_target(int width, int height, float sigma) {
  Mat g = Mat(height, width, CV_32F, 0.0f);
  g.at<float>(height/2, width/2) = 1.0f;

  Size zero(0, 0);
  GaussianBlur(g, g, zero, sigma);

  double minVal, maxVal;
  Point minPos, maxPos;
  minMaxLoc(g, &minVal, &maxVal, &minPos, &maxPos);
  g /= maxVal;

  int g_step = align(width * 2, 32);

  Mat G = Mat(height, width, CV_32FC2, allocate(height, width, g_step, 1), g_step * sizeof(float));
  dft(g, G, DFT_COMPLEX_OUTPUT);

  G.cols = G.cols / 2 + 1;
  return G;
}

Size Mosse::getWindowsCount() {
  return Size(
      frameSize.width * OVERLAP_DIVISOR / winSize.width - 1,
      frameSize.height * OVERLAP_DIVISOR / winSize.height - 1);
}

#ifdef CUDA
extern int max_size;
#else
int max_size;
#endif

Mosse::Mosse(int width, int height, int frame_width, int frame_height) {

  smoothLine = SmoothLine(10);
  width = getOptimalDFTSize(width);
  height = getOptimalDFTSize(height);
  winSize = Size(width, height);
  frameSize = Size(frame_width, frame_height);
  dbgidx = -1;


  int winStep = align(width, 32);
  float *winMemory = allocate(height, width, winStep, 1);
  window = Mat(height, width, CV_32FC1, winMemory, winStep * sizeof(float));
  createHanningWindow(window, winSize, CV_32FC1);

  G = gen_corr_target(width, height, 0.8);

  auto totSize = getWindowsCount();
  int batch_size = max_windows = max_size = totSize.width * totSize.height;

  Mat **hMats[3] = { &winH, &h1Tmp, &h2Tmp };

  // Initialization of winH, h1Tmp, h2Tmp
  for (int i = 0; i < 3; i++) {
    *(hMats[i]) = new Mat[batch_size];
    int winh_step = align(width * 2, 32);
    float *winHMemory = allocate(height, width, winh_step, batch_size);
    for (int j = 0; j < batch_size; j++) {
      int batch_stride = winh_step * height;
      (*(hMats[i]))[j] = Mat(height, width / 2 + 1, CV_32FC2, winHMemory + batch_stride*j, winh_step * sizeof(float));
    }
  }

  fft_in = new Mat[batch_size];
  fft_out = new Mat[batch_size];

  int step = align(width, 64); //TODO: Debug CUDA, it fails when alignment=32!!!

  float *inptr = allocate(height, width, step, batch_size);
  float *outptr = allocate(height, width, step * 2, batch_size);

  for (int i = 0; i < batch_size; i++) {
    int in_batch_stride = height * step;
    int out_batch_stride = height * step * 2;
    fft_in[i] = Mat(height, width, CV_32FC1, inptr + in_batch_stride*i, step * sizeof(float));
    fft_out[i] = Mat(height, width/2 + 1, CV_32FC2, outptr + out_batch_stride*i, step * 2 * sizeof(float));
  }

#ifdef CUDA
  if (use_gpu) {
    fft = new FFT(FFT_CUFFT, DIR_FORWARD, fft_in[0], fft_out[0], batch_size);
    fft_back = new FFT(FFT_CUFFT, DIR_BACKWARD, fft_out[0], fft_in[0], batch_size);
  }
  else
#endif
  {
    fft = new FFT(FFT_FFTW, DIR_FORWARD, fft_in[0], fft_out[0], batch_size);
    fft_back = new FFT(FFT_FFTW, DIR_BACKWARD, fft_out[0], fft_in[0], batch_size);
  }
}

void Mosse::divSpec(Mat const& A, Mat const& B, Mat &C) {
  for(int i = 0; i < A.rows; i++){
    const complex<float> *rowa = A.ptr<complex<float> >(i);
    const complex<float> *rowb = B.ptr<complex<float> >(i);
    complex<float> *rowc = C.ptr<complex<float> >(i);
    for(int j = 0;j < A.cols; j++){
        rowc[j] = conj(rowa[j] / rowb[j]);
    }
  }
}


void mulSpec(Mat const& A, Mat const& B, Mat &C) {
  for(int i = 0; i < A.rows; i++){
    const complex<float> *rowa = A.ptr<complex<float> >(i);
    const complex<float> *rowb = B.ptr<complex<float> >(i);
    complex<float> *rowc = C.ptr<complex<float> >(i);
    for(int j = 0;j < A.cols; j++){
        rowc[j] = rowa[j] * conj(rowb[j]);
    }
  }
}

void Mosse::updateKernel(MosseWindow &win) {
  divSpec(win.H1, win.H2, win.H);
}

inline void Mosse::initMosse(MosseWindow &win) {
  win.training_left = TRAINING_TIME;
  win.notgood_time = 0;
  win.alive_time = 0;
  win.perm_deleted = false;
  win.frame_good = true;
  win.init_status = MINIT_UNINITIALIZED;
  win.begin_position = win.position;
  Matx13f winBegPosition(win.begin_position.x, win.begin_position.y, 1);
  winBegPosition = winBegPosition * rot_matrix.trmat.t();
  win.position = Point(winBegPosition(0, 0), winBegPosition(0, 1));

  win.displacement = Point(0, 0);
  // win.H = Mat(winSize.height, winSize.width / 2 + 1, CV_32FC2);
  // Mat patch;
  // extractSubMat(frame, winSize, win.position + displacement, patch);
  // ess(patch, window);
  // Mat P;
  // dft(patch, P, DFT_COMPLEX_OUTPUT);
  // mulSpectrums(G, P, win.H1, 0, true);
  // mulSpectrums(P, P, win.H2, 0, true);
  // updateKernel(win);
#ifdef CUDA
    if (use_gpu)
      cudaDeviceSynchronize();
#endif
    win.H1.setTo(0);
    win.H2.setTo(0);
    win.H.setTo(0);
    win.init_status = MINIT_RESETTED;
}

Mat totframe;

int Mosse::updateWindow(Mat &frame, MosseWindow &win, float psr_th, int state, float rate) {

BEGIN_YIELD(state);
  win.alive_time++;
  // Update counters, with data based on previous frame
  if (win.now_outside)
    win.outside_time++;
  else
    win.outside_time = 0;

  if (!win.frame_good) {
    win.notgood_time++;
    if (win.notgood_time > 3) {
      win.mean_goodtime = 0;
    }
    win.good_time = 0;
  }
  else {
    win.good_time++;
    win.mean_goodtime++;// = max(win.mean_goodtime, win.good_time);
    win.notgood_time = 0;
  }

  if (win.notgood_time > 10) {
    win.perm_deleted = true;
  }

  Matx13f winPosition(win.position.x, win.position.y, 1);
  Matx13f winBegPosition(win.begin_position.x, win.begin_position.y, 1);

  winPosition = winPosition * rot_matrix.trinv.t();

  bool outside = false;
  //-winSize.height/OVERLAP_DIVISOR winSize.height/OVERLAP_DIVISOR
  if (winPosition(0, 0) >= frameSize.width || winPosition(0, 1) >= frameSize.height || winPosition(0, 1) <= 0 || winPosition(0, 0) <= 0 ) {
    // cout << "Deleted: " << win.position << endl;
    if (winPosition(0, 0) >= frameSize.width)
      winPosition(0, 0) -= frameSize.width;
    if (winPosition(0, 1) >= frameSize.height-winSize.height/OVERLAP_DIVISOR)
      winPosition(0, 1) -= frameSize.height-(2*winSize.height/OVERLAP_DIVISOR);

    if (winPosition(0, 0) < 0)
      winPosition(0, 0) += frameSize.width;
    if (winPosition(0, 1) < 0)
      winPosition(0, 1) += frameSize.height-(2*winSize.height/OVERLAP_DIVISOR);

    outside = true;
  }
  win.now_outside = outside;


  // bool outside = false;
  // if (winBegPosition(0, 0) >= frameSize.width || winBegPosition(0, 1) > frameSize.height-winSize.height/OVERLAP_DIVISOR || winBegPosition(0, 1) < winSize.height/OVERLAP_DIVISOR || winBegPosition(0, 0) <= 0 ) {
  //   win.deltime = MAX_DELTIME;
  //   win.good = false;
  //   outside = true;
  //   return;
  // }

  if (false && win.perm_deleted) { // Reinitialize TODO

    // // initMosse(frame, index, win); // Reset window
    // auto new_begin = winPosition;//Matx13f(win.begin_position.x, win.begin_position.y, 1);
    // // new_begin = new_begin * rot_matrix.trmat.t();
    //
    // if (outside) {
    //   win.reInit = 0;
    //   return -1;
    // }
    // else {
    //   if (win.training < 0) {
    //     win.reInit += win.training;
    //   }
    //   else {
    //     win.reInit += -5;
    //   }
    // }
    //
    // win.position = Point(new_begin(0, 0), new_begin(0, 1));//win.begin_position;
    // win.good = true;
    // win.deltime = 0;
    //
    // win.reInit = max(win.reInit, -200);
    // // printf("Reinit: %d\n", win.reInit);
    // win.training = TRAINING_TIME;
    // return -1;
  }

  // win.dbgPatch = patch;
  if (win.index == dbgidx) {
    Mat tmp; dft(win.H, tmp, DFT_REAL_OUTPUT | DFT_INVERSE);
    imshow("DbgPatch", tmp);
  }

  if (!use_gpu) {
    Mat &patch = fft_in[win.index];
    extractSubMatTransform(frame, winSize, win.position, patch, trMat);
    preprocess(patch, window);
  }

  DO_FFT(MSTAGE1_FFT);

  if (!win.should_evaluate()) return -1;

  if (!use_gpu)
    convolve_step(win);

  DO_FFT_INV(MSTAGE2_FFT_INV);

  // if (win.training) {
  //   rate = 0.175;
  // }
  // else {
  //   rate = 0.125;
  // }

  // if (!win.good) return -1; TORESTORE

  Mat patch = fft_in[win.index];
  if (win.should_evaluate()) {
    double psr = correlate(patch, win);

    win.frame_good = psr > psr_th;//  && (norm(win.delta) < 15);

    // int isgood = 0;
    // int total = 0;
    // for (int i = 0; i < 4; i++) {
    //   if (win.neighb[i]) {
    //     total++;
    //     if (win.neighb[i]->notgood_time == 0) {
    //       Point dist = win.position - win.neighb[i]->position;
    //       float totdist = norm(dist);
    //       Point expdist = Point(winSize.width/OVERLAP_DIVISOR, winSize.height/OVERLAP_DIVISOR);
    //       float totexpdist = norm(expdist);
    //       if (totdist - totexpdist < 2*totexpdist && (abs(dist.y-expdist.y) < expdist.y*1.3))
    //         isgood++;
    //     }
    //   }
    // }
    // if (total > 0 && isgood == 0)
    //   win.frame_good = false;

    // win.good = true;

    // if (!win.frame_good && win.training <= 0) { TORESTORE
    //   win.deltime++;
      // if (win.deltime >= MAX_DELTIME) return -1; TORESTORE
    // }
    // else { TORESTORE
    //   win.deltime = 0;
    // }

    win.training_left--;
    if (win.perm_deleted || win.now_outside) return -1;

    win.displacement += win.delta;
    win.position += win.delta;
  }

  if (!use_gpu) {
    extractSubMatTransform(frame, winSize, win.position, patch, trMat);
    preprocess(patch, window);
  }

  DO_FFT(MSTAGE3_FFT);

  if (!use_gpu) {

    float rate = 0.125;
    // if (win.training > 0) {
    //   rate = 0.175;
    // }
    // else {
    //   // rate = 0.125;
    //   rate = 0.100;
    // }
    Mat A = fft_out[win.index], h1, h2;

    mulSpectrums(G, A, h1, 0, true);
    mulSpectrums(A, A, h2, 0, true);

    win.H1 = win.H1 * (1.0-rate) + h1 * rate;
    win.H2 = win.H2 * (1.0-rate) + h2 * rate;
    updateKernel(win);
  }
END_YIELD();
}

#ifdef CUDA
const cuda_funcs::PatchDesc *patches;
#endif

double Mosse::convolve_step(MosseWindow &win) {
  Mat &P = fft_out[win.index];
  // dft(patch, P, DFT_COMPLEX_OUTPUT);
  mulSpectrums(P, win.H, P, 0, true);

  if (!this->fft->normalized) {
    win.H /= P.rows * P.cols;
  }
}

double Mosse::correlate(Mat &patch, MosseWindow &win) {
  int mx, my;
  double smean, sstd, maxVal;

  if (win.init_status == MINIT_RESETTED) {
    win.delta = Point(0, 0);
    win.init_status = MINIT_INITIALIZED;
    return +INFINITY;
  }

  if (!use_gpu) {
    Mat resp = fft_in[win.index], side_resp;

    double minVal;
    Point minPos, maxPos;
    minMaxLoc(resp, &minVal, &maxVal, &minPos, &maxPos);

    mx = maxPos.x;
    my = maxPos.y;


    side_resp = resp.clone();

    Point beg(mx-5, my-5);
    Point end(mx+5, my+5);

    rectangle(side_resp, beg, end, 0, -1);
    Scalar _smean, _sstd;
    meanStdDev(side_resp, _smean, _sstd);
    smean = _smean[0];
    sstd = _sstd[0];
  }
#ifdef CUDA
  else {
    int totPixels = patch.rows * patch.cols;
    auto &crt = patches[win.index];

    maxVal = crt.maxValue;
    if (dbgidx == win.index) {
      cout << "Coords: " << crt.coords << endl;
    }

    smean = crt.sum/totPixels;
    float sqmean = crt.sqsum / totPixels;
    sstd = sqrt(sqmean-smean*smean);
    int row_stride = patch.step / sizeof(float);
    mx = crt.coords % row_stride;
    my = crt.coords / row_stride;

    // printf("Patch: %d Max: (%f, %f) MaxLoc: (%d, %d) / (%d, %d) Std: (%f, %f)\n", index, crt.maxValue, maxVal, crt.coords/row_stride, crt.coords%row_stride,
    //   maxPos.y, maxPos.x, std, sstd[0]);
  }
#endif

  float eps = 1e-5;
  double psr = (maxVal-smean) / (sstd+eps);

  win.delta = Point(mx - winSize.width/2 - 1/* to avoid right translation of windows (why?) */, my - winSize.height/2);
  //cout << "PSR:: " << psr << endl;
  return psr;
}

static TransformMat gmat;

static void circle(Mat &frame, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0) {
  auto tpoint = (Matx13f(center.x, center.y, 1) * gmat.trinv.t()).get_minor<1, 2>(0, 0);
  cv::circle(frame, Point(tpoint(0, 0), tpoint(0, 1)), radius, color, thickness, lineType, shift);
}

void Mosse::reset(cv::Mat mask, TransformMat trmat) {
  for (auto win : mosseWindows) {
    allocator.deallocate(win);
  }
  mosseWindows.clear();
  refresh(mask, trmat);
}

void Mosse::refresh(Mat mask, TransformMat trmat) {

  this->rot_matrix = trmat;
  // this->trMat = trmat.trinv;
  // this->trMat -= Matx33f(0, 0, 0, 0, 0, 0, displacement.x, displacement.y, 0);
  // this->trMat = Matx33f(1, 0, 0, 0, 1, 0, displacement.x, displacement.y, 1);
  this->trMat = trmat.trinv.t();
  // this->trMat -= Matx33f(1, 0, 0, 0, 1, 0, 0, 0, 1);
  // Mat grayscale, gsfloat;
  // cvtColor(frame, grayscale, COLOR_BGR2GRAY);
  // grayscale.convertTo(gsfloat, CV_32FC1);

  // for (auto win : mosseWindows) {
  //   allocator.deallocate(win);
  // }
  // mosseWindows.clear();

  for (auto &win : mosseWindows) {
    if (mask.at<float>(trpoint(trmat, win->position, true)) > 0.5f) {
      allocator.deallocate(win);
      win = nullptr;
      continue;
    }
    circle(mask, win->position, winSize.width/2/OVERLAP_DIVISOR, Scalar(255, 255, 255), -1);
  }

  int nidx = mosseWindows.size()-1;
  for (int i = 0; i < nidx;) {
    if (!mosseWindows[i]) {
      swap(mosseWindows[i], mosseWindows[nidx--]);
    }
    else
      i++;
  }
  while (mosseWindows.size() && !mosseWindows.back()) mosseWindows.pop_back();

  Size winsCount = getWindowsCount();
  vector<MosseWindow*> wins[winsCount.width * winsCount.height];

  // #pragma omp parallel for
  // for (int i = 0; i < winsCount.width; i++) {
  //   for (int j = 0; j < winsCount.height; j++) {
  //     Point pos = Point((i+1) * winSize.width/OVERLAP_DIVISOR, (j+1) * winSize.height/OVERLAP_DIVISOR);
  //     if (cnt > 1 && mask.at<Vec3b>(pos)!=Vec3b(0, 0, 0)) {
  //       continue;
  //     }
  //     MosseWindow &tmp = *allocator.allocate();
  //     // tmp.index = i * winsCount.height + j;
  //     tmp.H = winH[tmp.index];
  //     tmp.H1 = h1Tmp[tmp.index];
  //     tmp.H2 = h2Tmp[tmp.index];
  //     printf("Index: %d\n", tmp.index);
  //
  //     tmp.position = pos;
  //     initMosse(tmp);
  //     wins[i].push_back(&tmp);
  //   }
  // }
  for (auto &win : mosseWindows) {
    circle(mask, win->position, winSize.width/OVERLAP_DIVISOR, Scalar(255, 255, 255), -1);
  }

  for (int i = winSize.height/2; i < mask.rows-winSize.height/2; i++) {
    for (int j = winSize.width/2; j < mask.cols-winSize.width/2; j++) {
      Point pos = Point(j, i);
      if (mask.at<float>(pos) > 0.5f) {
        continue;
      }
      addWindow(pos);
      cv::circle(mask, pos, winSize.width/OVERLAP_DIVISOR, Scalar(255, 255, 255), -1);
    }
  }


  for (auto &vec : wins) {
    mosseWindows.insert(
      mosseWindows.end(),
      std::make_move_iterator(vec.begin()),
      std::make_move_iterator(vec.end())
    );
  }
}

static bool isInsideImage(TransformMat const& trmat, Point p, Size winSize, Size frameSize) {
  Matx13f pt(p.x, p.y, 1);
  pt = pt * trmat.trinv.t();
  if (pt(0, 0) < winSize.width/2 ||
      pt(0, 1) < winSize.height/2 ||
      pt(0, 0) > frameSize.width-winSize.width/2 ||
      pt(0, 1) > frameSize.height-winSize.height/2)
    return false;
  return true;
}

static vector<Point> ptsb;
static DistanceEstimator est;


static void polylines(Mat& frame, const vector<Point> &pts, bool isClosed, const Scalar& color, int thickness=1, int lineType=8, int shift=0) {
  ptsb.resize(pts.size());

  for (int i = 0; i < pts.size(); i++) {
    auto tpoint = (Matx13f(pts[i].x, pts[i].y, 1) * gmat.trinv.t()).get_minor<1, 2>(0, 0);
    ptsb[i] = Point(tpoint(0, 0), tpoint(0, 1));
  }
  cv::polylines(frame, ptsb, isClosed, color, thickness, lineType, shift);

  // est.reset();
  // for (int i = 0; i < mosseWindows.size(); i++) {
  //   est.addPoint(mosseWindows[i]->position);
  // }
}


Mat Mosse::update(Mat frame, Mat outframe, float psr_th, TransformMat trmat) {
    static int min_idx = 0;
    this->rot_matrix = trmat;
    gmat = trmat;
    this->trMat = trmat.trinv.t();

    Mat grayscale, fft;

    int siz = mosseWindows.size();

    for (int stage = MINIT; stage <= MSTAGE3_FFT; ++stage) {
      switch (stage) {
        case MSTAGE1_FFT:
        case MSTAGE3_FFT:
#ifdef CUDA
          if (use_gpu)
            cuda_funcs::preprocess(mosseWindows, winSize, trMat, frame, fft_in[0], window);
#endif
          this->fft->execute();
          break;
        case MSTAGE2_FFT_INV:
#ifdef CUDA
          if (use_gpu)
            cuda_funcs::convolve_cmplx(fft_out[0], winH[0], fft_out[0], max_windows);
#endif
          this->fft_back->execute();
#ifdef CUDA
          cudaDeviceSynchronize();
#endif
          // imshow("Mosse", mosseWindows[min_idx`]->dbgPatch);
          if (dbgidx >= 0)
            imshow("MosseFFT", fft_in[dbgidx] / 256.0f);
#ifdef CUDA
          if (use_gpu)
            patches = cuda_funcs::max_std_process(fft_in[0], max_windows);
#endif
          break;
      }
#ifdef CUDA
      cudaDeviceSynchronize();
#endif
      if (!(use_gpu && ((stage == MSTAGE1_FFT) || (stage == MSTAGE3_FFT) ))) {
        #pragma omp parallel for if (!use_gpu)
        for (int i = 0; i < siz; i++) {
          updateWindow(frame, *(mosseWindows[i]), psr_th, stage);
        }
      }
    }

#ifdef CUDA
    if (use_gpu) {
      cuda_funcs::convolve_cmplx(G, fft_out[0], h1Tmp[0], max_windows, cuda_funcs::OP_UPD_CORR_G);
      cuda_funcs::convolve_cmplx(fft_out[0], fft_out[0], h2Tmp[0], max_windows, cuda_funcs::OP_UPD_CORR);
      cuda_funcs::convolve_cmplx(h1Tmp[0], h2Tmp[0], winH[0], max_windows, cuda_funcs::OP_DIV);
    }
#endif
    // first1 = true;

    vector<Point> pts;

    for (auto& win : mosseWindows) {
      if (win->perm_deleted) {
        Scalar color(16, 16, 255, 48);
        circle(outframe, win->position, 4, color, -1);
        allocator.deallocate(win);
        win = NULL;
      }
    }

    int nidx = mosseWindows.size()-1;
    for (int i = 0; i < nidx;) {
      if (!mosseWindows[i]) {
        swap(mosseWindows[i], mosseWindows[nidx--]);
      }
      else
        i++;
    }
    while (mosseWindows.size() && !mosseWindows.back()) mosseWindows.pop_back();
    // random_shuffle(mosseWindows.begin(), mosseWindows.end());
    // if (mosseWindows.size() > 200) mosseWindows.pop_back();


    // vector<MosseWindow*> newWindows;
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    // imshow("Mosse", mosseWindows[min_idx]->dbgPatch);

    int min_val = 0;
    int tmp = 0;
    for (auto win : mosseWindows) {
      if (win->perm_deleted)
        continue;
      Scalar color(255, 255, 255, 64);
      Scalar hcolor(0, 255, 0, 128);

      double alpha = min((15.0-win->mean_goodtime)/15.0, 1.0);

      circle(outframe, win->position, 4, hcolor * alpha + color * (1.0 - alpha), -1);

      if (win->mean_goodtime > 15  /*&& win->training_left < -5*/) {//(win->good && (win->reInit ? (win->training < win->reInit-5) : (win->training < -5))) {
        circle(outframe, win->position, 16, color, -1);
        pts.push_back(win->position);
        if (min_val < win->position.y) {
          min_val = win->position.y;
          min_idx = tmp;
        }
      }
      tmp++;
      // if (win->deltime < MAX_DELTIME || win->good) {
      //   newWindows.push_back(win);
      // }
    }

    for (auto win : mosseWindows) {
      Scalar icolor(255, 0, 0, 128);
      if (win->mean_goodtime > 15 && !win->perm_deleted)
        circle(outframe, win->position, 3, icolor, -1);
    }

    // for (int i = 0; i < mosseWindows.size(); i++) {
    //   if (mosseWindows[i]->good_time > 10 && mosseWindows[i]->training_left < -5)
    //     est.updatePoint(i, mosseWindows[i]->position);
    //   else
    //     est.delPoint(i);
    // }
    // est.draw(outframe);

    // Matx13f test(frameSize.width/2, frameSize.height/2, 1);
    // test = test * rot_matrix.trmat.t();
    // cout << winBegPosition << endl;
    // circle(outframe, Point(test(0, 0), test(0, 1)), 4, Scalar(255, 255, 255), -1);
    // mosseWindows = newWindows;
    // random_shuffle(mosseWindows.begin(), mosseWindows.end());

    vector<Point> hull;
    vector<Point> horizline;
    /*if (pts.size() > 4) {
      convexHull(pts, hull);
    }

    for (int i = 0; i < (int)hull.size()-1; i++) {
      if (hull[i].x > hull[i+1].x) {
        horizline.push_back(hull[i]);
      }
    }*/
    int hoffset = 10;


    if (mosseWindows.size()) {

      int minidx = -1;
      for (int i = 0; i < mosseWindows.size(); i++) {
        if (mosseWindows[i]->frame_good && (minidx == -1 || mosseWindows[minidx]->position.y < mosseWindows[i]->position.y)) {
          minidx = i;
        }
      }

      // if (minidx >= 0)
      // {
      //   Mat test = mosseWindows[minidx]->dbgPatch;
      //   double minVal, maxVal;
      //   Point minPos, maxPos;
      //   minMaxLoc(test, &minVal, &maxVal, &minPos, &maxPos);
      //   imshow("Test1", (test-minVal) / (maxVal-minVal));
      // }

      sort(pts.begin(), pts.end(), [](Point const& a, Point const& b){ return a.x < b.x; });

      int deldist = 32;
      int notins_dist = 10;

      for (Point p : pts) {

        int delidx = horizline.size();
        for (int i = horizline.size()-1; i >= 0; i--) {
          if (p.x - horizline[i].x > deldist) break;
          if (horizline[i].y < p.y) {
            delidx = i;
          }
          else {
            break;
          }
        }
        while (horizline.size() > delidx) {
          horizline.pop_back();
        }
        if (!horizline.size() || (p.y > horizline.back().y) || (p.x - horizline.back().x > notins_dist) ) {
          if (isInsideImage(trmat, p, winSize, Size(frame.cols, frame.rows)))
            horizline.push_back(p);
        }
      }

      if (horizline.size() > 0) {

        for (auto p : horizline) {
          Scalar color(0, 200, 255, 255);
          circle(outframe, p, 7, color, -1);
        }

        // horizline.pop_back();
        Scalar hcolor(255, 255, 0, 255);
        smoothLine.addLine(horizline, trmat);
        smoothLine.drawLine(outframe, trmat);
        // polylines(outframe, horizline, false, hcolor, 3);

        // int width = outframe.size().width;

        // Point fill1 = horizline[0];
        //
        // Point fill2 = horizline.back();
        // fill2.x = width-1;

      //   vector<Point> fill_hline;
      //   int ptr = 0;
      //   for (int i = -hoffset; i < width + hoffset; i++) {
      //     while (ptr < horizline.size() && horizline[ptr].x <= i) ptr++;
      //
      //     if (ptr == 0) {
      //       fill_hline.push_back(Point(i, -100000));//horizline[ptr].y));// + horizline[ptr]);
      //       continue;
      //     }
      //
      //     float alpha = (!(horizline[ptr].x - horizline[ptr-1].x)) ? 1 : ((float)i-horizline[ptr-1].x) / ((float)horizline[ptr].x - horizline[ptr-1].x);
      //     if (alpha > 1)
      //       alpha = -100000;
      //     fill_hline.push_back(Point(i, horizline[ptr-1].y * (1-alpha) + horizline[ptr].y * (alpha) ) );// + horizline[ptr]);
      //   }
      //   horizhistory.push_back(fill_hline);
      //
      //   if (horizhistory.size() > 10) {
      //     horizhistory.erase(horizhistory.begin());
      //   }
      // }
    }

    // if (horizhistory.size()) {
    //   vector<Point> final_hline(outframe.size().width+hoffset*2);
    //
    //   for (int i = 0; i < horizhistory.back().size(); i++) {
    //     final_hline[i].x = horizhistory.back()[i].x;
    //   }
    //
    //   for (auto hline : horizhistory) {
    //
    //       for (int i = 0; i < hline.size(); i++) {
    //         final_hline[i].y += hline[i].y;
    //       }
    //
    //   }
    //
    //   vector<Point> roi_hline;
    //   Size outsize = outframe.size();
    //   for (auto &el : final_hline) {
    //     el.y /= (int)horizhistory.size();
    //
    //     if (el.x < 0 || el.x > outsize.width || el.y < 0 || el.y > outsize.height)
    //       continue;
    //     roi_hline.push_back(el);
    //   }
    //   Scalar hcolor(255, 255, 0, 255);
    //   polylines(outframe, roi_hline, false, hcolor, 3);
    // }

    return outframe;
  }
}

void Mosse::addWindow(Point coords) {
  if (mosseWindows.size() >= max_windows) return;
  auto& win = *allocator.allocate();
  win.H = winH[win.index];
  win.H1 = h1Tmp[win.index];
  win.H2 = h2Tmp[win.index];
  win.position = coords;
  initMosse(win);
  mosseWindows.push_back(&win);
}

void Mosse::reset_history() {
  horizhistory.clear();
}

void HorizonLine::add(vector<Point> const& horizline, Matx33f local) {
  local = local.inv();

}

void HorizonLine::draw(Mat &frame, int startidx) {

}

void HorizonLine::setg(cv::Matx33f global) {
  this->global = global;
}

void Mosse::displaydbg(cv::Point coords) {
  auto tpoint = (Matx13f(coords.x, coords.y, 1) * gmat.trmat.t()).get_minor<1, 2>(0, 0);
  coords = Point(tpoint(0, 0), tpoint(0, 1));
  for (auto &win : mosseWindows) {
    if (norm(win->position-coords) < 5) {
      dbgidx = distance(mosseWindows.data(), &win);
      printf("Debug = %d\n", dbgidx);
      break;
    }
  }
}

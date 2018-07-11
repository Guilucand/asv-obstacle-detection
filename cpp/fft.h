#include <fftw3.h>

#ifdef CUDA
#include <cufft.h>
#endif

enum FFT_MODE {
  FFT_FFTW,
  FFT_OPENCV,
  FFT_OPENCV_PARALLEL,
  FFT_OPENCV_CUDA,
  FFT_CUFFT,
};

enum FFT_DIRECTION {
  DIR_FORWARD,
  DIR_BACKWARD
};


class FFT {
public:
  FFT(FFT_MODE mode, FFT_DIRECTION dir, cv::Mat source, cv::Mat dest, int count = 1);
  void execute();

  cv::Mat source_;
  cv::Mat dest_;

  bool normalized;

private:
  FFT_MODE mode_;
  FFT_DIRECTION dir_;

  int count_;

  union {

#ifdef CUDA
    cufftHandle cudaHandle;
#endif
    fftwf_plan fftwPlan;
  };

};

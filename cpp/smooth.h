#include "image_stab.h"

class SmoothLine {

public:
  SmoothLine(int smoothness = 0);

  void addLine(std::vector<cv::Point> points, TransformMat trmat);
  void drawLine(cv::Mat frame, TransformMat trmat);

private:
  int smoothness;
  std::deque<std::vector<cv::Point> > lines;
};

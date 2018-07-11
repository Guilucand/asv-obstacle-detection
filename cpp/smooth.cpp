#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "smooth.h"
#include "util.h"

using namespace std;
using namespace cv;


SmoothLine::SmoothLine(int smoothness) {
  this->smoothness = smoothness=10;
}

void SmoothLine::addLine(std::vector<cv::Point> points, TransformMat trmat) {


  sort(points.begin(), points.end(), [&](Point const& a, Point const& b){return a.x < b.x;});
  lines.push_back(points);

  while (lines.size() > smoothness) lines.pop_front();
}

void SmoothLine::drawLine(Mat frame, TransformMat trmat) {

  vector<int> idxs;
  idxs.resize(lines.size());
  fill(idxs.begin(), idxs.end(), 0);

  vector<Point> line;

  for (int x = 0; x < frame.cols; x+=2) {

    // Update indices
    for (int i = 0; i < lines.size(); i++) {
      while ((idxs[i] < lines[i].size()) && (trpoint(trmat, lines[i][idxs[i]], true).x < x))
       idxs[i]++;
    }

    float value = 0;
    int count = 0;
    for (int i = 0; i < lines.size(); i++) {
      if (!idxs[i] || idxs[i] >= lines[i].size()) continue;
      auto prev = trpoint(trmat, lines[i][idxs[i]-1], true);
      auto next = trpoint(trmat, lines[i][idxs[i]], true);

      int dp = x - prev.x;
      int dn = next.x - x;
      dp *= dp;
      dn *= dn;

      value += static_cast<float>(prev.y * dn + next.y * dp) / (dp+dn);
      count++;
    }
    if (count > 0)
      line.push_back(Point(x, value / count));
  }
  // cout << "Vector: ";
  // for (auto v : line) cout << v << " ";
  // cout << endl;
  Scalar hcolor(255, 255, 0, 255);


  cv::polylines(frame, line, false, hcolor, 3);
}

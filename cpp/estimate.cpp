#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <numeric>

#include "estimate.h"


using namespace cv;
using namespace std;


int DistanceEstimator::addPoint(Point point) {
  points[gidx].push_back(point);
  return gidx;
}

void DistanceEstimator::updatePoint(int idx, cv::Point point) {
  points[idx].push_back(point);
}

void DistanceEstimator::delPoint(int idx) {
  points[idx].clear();
}

void DistanceEstimator::draw(cv::Mat frame) {
  vector<pair<float, int> > dists;

  for (auto &p : points) {
    if (p.second.size()) {
      vector<float> distsp;
      for (auto &p2 : points) {
        if (p2.second.size())
          distsp.push_back(norm(p.second.back()-p2.second.back()));// - norm(p.second.front()-p2.second.front()));
      }
      int cnt = min((int)distsp.size(), 10);
      if (cnt < 7) continue;
      nth_element(distsp.begin(), distsp.begin() + cnt, distsp.end());
      float sum = std::accumulate(distsp.begin(), distsp.begin() + cnt, 0);

      // if (p.second.size() < 5)
      //   dists.push_back(make_pair(-norm(p.second.back()-p.second.front()), p.first));
      // else
      //   dists.push_back(make_pair(-norm(p.second.back()-p.second[p.second.size()-5]), p.first));
      dists.push_back(make_pair(-sum, p.first));
    }
  }

  sort(dists.begin(), dists.end());

  int id = 0;
  for (auto &pt : dists) {
    Scalar hcolor(0, 0, 255 * (1-((float)id)/dists.size()), 255);
    // if (120 <= id && id <= 150)
    if (id > 20) break;
      circle(frame, points[pt.second].back(), 4, hcolor, -1);
    id++;
  }

}

void DistanceEstimator::reset() {
  points.clear();
  gidx = 0;
}

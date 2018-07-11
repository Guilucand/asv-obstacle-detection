


class DistanceEstimator {
public:
  int addPoint(cv::Point point);
  void updatePoint(int idx, cv::Point point);
  void delPoint(int idx);
  void draw(cv::Mat frame);
  void reset();

private:
  std::map<int, std::vector<cv::Point> > points;
  int gidx = 0;
};

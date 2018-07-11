#pragma once

struct TrackHistory {
  cv::Point_<float> first;
  cv::Point_<float> prev;
  int duration;
};

struct TransformMat {
  TransformMat();
  cv::Matx33f trmat;
  cv::Matx33f trinv;
};

class ImgStab {
private:
  int track_len = 10;
  int detect_interval = 5;
  std::vector<cv::Point_<float> > tracks;
  std::vector<TrackHistory> tracks_hist;
  std::vector<unsigned char> status;
  std::vector<unsigned char> status2;
  std::vector<cv::Point_<float> > inv_tracks;
  std::vector<cv::Point_<float> > out_tracks;
  cv::Mat prev_frame;
  cv::Mat mask;
  cv::Matx33f X;
  int frame_idx = 0;

  void find_matrix(cv::Mat const& frame);
  void compute_optflow(cv::Mat const& frame);
  void find_points(cv::Mat const& frame);
public:
  ImgStab();
  void update(cv::Mat const& frame);
  void mat_update(TransformMat &mats);
  cv::Point ftransform(TransformMat &mats, cv::Mat &frame, bool constant = true, int debug = 0, bool onlyRotation = false, cv::Point displacement = cv::Point(0, 0), bool inverse = false);
  // void itransform(TransformMat &mats, cv::Mat const& frame, bool constant = true);
  void reset(TransformMat &mats);
  void reset_all();


};

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#ifdef CUDA

#ifdef JETSON
#include <opencv2/gpu/gpu.hpp>
#else
#include <opencv2/core/cuda.hpp>
#endif

#endif

#include "image_stab.h"

using namespace std;
using namespace cv;

TransformMat::TransformMat() {
  trmat = Matx33f (1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
  trinv = Matx33f (1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
}

ImgStab::ImgStab() {
  X = Matx33f (1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
  // tracks.push_back({30, 30});
  //
  // for (int i = 0; i < 384; i++)
  //   for (int j = 0; j < 216; j++)
  //   tracks.push_back({i, j});
}


void ImgStab::update(Mat const& frame) {

  Mat tmp_frame;
  frame.copyTo(tmp_frame);
  if (!prev_frame.empty() && tracks.size() > 4) {
    compute_optflow(frame);
    find_matrix(frame);
  }


  if (frame_idx % detect_interval == 0 || tracks.size() < 20) {
      find_points(tmp_frame);
  }
  frame_idx++;

  tmp_frame.copyTo(prev_frame);

  for (int i = 0; i < tracks.size(); i++) {
    tracks_hist[i].prev = tracks[i];
    tracks_hist[i].duration++;
  }

}

void ImgStab::mat_update(TransformMat &mats) {
  mats.trmat = mats.trmat * X;
  mats.trinv = mats.trmat.inv();
}

Point ImgStab::ftransform(TransformMat &mats, Mat &frame, bool constant, int debug, bool onlyRotation, Point displacement, bool inverse) {

  if (debug) {
    Scalar colors[] = {
      Scalar(0, 0, 0),
      Scalar(0, 0, 255),
      Scalar(0, 255, 255),
      Scalar(0, 255, 0),
      Scalar(255, 0, 0),
    };

    for (auto &track : tracks_hist) {
      int dur = min(track.duration, 49);
      switch (debug) {
        case 1:
          circle(frame, track.prev, 3, colors[dur/10], -1);
          break;
        case 2:
          if (dur > 30)
            circle(frame, track.prev, 2, Scalar(255, 255, 255), -1);
          break;
      }
    }
  }
  auto mat = mats.trmat;
  Matx13f diff(0, 0, 1);
  if (onlyRotation) {
    Size size = frame.size();

    auto center = Matx13f(size.width/2 - displacement.x, size.height/2 - displacement.y, 1) * mat.t();
    diff = center - Matx13f(size.width/2, size.height/2, 1);
    // cout << "Center: " << center << endl;
    // cout << "Diff: " << diff << endl;
    // cout << "Mat: " << mat << endl;
    mat(0, 2) -= diff(0, 0);
    mat(1, 2) -= diff(0, 1);
  }


  if (inverse)
    mat = mat.inv();

  warpAffine(frame, frame, mat.get_minor<2, 3>(0, 0), Size(frame.cols, frame.rows), INTER_LINEAR, constant ? BORDER_CONSTANT : BORDER_REPLICATE);
  return Point(-diff(0, 0), -diff(0, 1));
}

void ImgStab::find_matrix(Mat const& frame) {
  // printf("Entering!!\n");
  vector<Point_<float> > prev_pts;
  vector<Point_<float> > next_pts;

  vector<Point_<float> > tmp_pts;
  for (auto &hist : tracks_hist) {
    tmp_pts.push_back(hist.prev);
  }

  // Matx23f m(trmat(0, 0), trmat(0, 1), trmat(0, 2),
  //           trmat(1, 0), trmat(1, 1), trmat(1, 2));
  // m.row(0) = trmat.row(0);
  // m.row(1) = trmat.row(1);
  //
  // transform(tmp_pts, prev_pts, m);
  prev_pts = tmp_pts;
  // transform(tracks, next_pts, m);
  next_pts = tracks;
  vector<Point_<float> > trans_pts;

  int num_iters = 10;
  int min_points = 20;

  for (int i = 0; i < num_iters; i++) {
    vector<Scalar_<float> > M;
    vector<float> Y;
    vector<float> Xt(4);

    float avg = 0;
    float median = 0;
    if (i) {
      vector<float> weights;
      for (int j = 0; j < next_pts.size(); j++) {
        float weight = -norm(trans_pts[j] - prev_pts[j]);
        weights.push_back(weight);
        avg += weight;
      }
      nth_element(weights.begin(), weights.begin() + (next_pts.size()/32), weights.end());
      median = weights[next_pts.size() / 2];
      avg /= next_pts.size();
    }
    // printf("Median: %f size: %d\n", median, (int)next_pts.size());

    int pos = 0;
    for (int j = 0; j < next_pts.size(); j++) {
      auto &pt  = next_pts[j];
      float weight = 1.0f;
      if (i) {
        weight = -norm(trans_pts[j] - prev_pts[j]);
        weight = weight < median ? 0.6f : 1.0f;
      }
      weight *= min(30, tracks_hist[j].duration);
      if (weight > 0.1) pos++;
      M.push_back({ pt.y * weight, pt.x * weight, 1 * weight, 0 });
      M.push_back({ pt.x * weight, -pt.y * weight, 0, 1 * weight });
    }


    int incl = 0;
    for (int j = 0; j < prev_pts.size(); j++) {
      auto &pt  = prev_pts[j];
      float weight = 1.0f;
      if (i) {
        weight = -norm(trans_pts[j] - prev_pts[j]);
        weight = weight < median ? 0.6f : 1.0f;
        if ((i == (num_iters-1) || pos < min_points) && weight > 0.7) {
          // circle(const_cast<Mat&>(frame), tracks[j], 8, 0, -1);
          incl++;
        }
      }
      weight *= min(30, tracks_hist[j].duration);
      Y.push_back(pt.y * weight);
      Y.push_back(pt.x * weight);
    }
    // if (incl) {
    //   printf("Included: %d/%d\n", incl, (int)next_pts.size());
    // }
    if (i && pos < min_points) break;

    Mat mM = Mat(M.size(), 4, CV_32FC1, M.data(), sizeof(Scalar_<float>));
    Mat mY = Mat(Y.size(), 1, CV_32FC1, Y.data(), sizeof(float));
    Mat mXt = Mat(4, 1, CV_32FC1, Xt.data(), sizeof(float));

    try {
      solve(mM, mY, mXt, DECOMP_SVD);
    }
    catch(exception e) {
      frame_idx = 0;
      X = Matx33f(1, 0, 0,
                  0, 1, 0,
                  0, 0, 1);
      return;
    }


    float scale = sqrt(Xt[0]*Xt[0] + Xt[1] * Xt[1]);
    // cout << "Count: " << M.size() << endl;
    // cout << "Scale: " << scale << endl;
    // cout << mXt << endl;
    Xt[0] /= scale;
    Xt[1] /= scale;

    X = Matx33f(Xt[0], -Xt[1], Xt[3],
                        Xt[1],  Xt[0], Xt[2],
                        0, 0, 1);
    transform(next_pts, trans_pts, X.get_minor<2, 3>(0, 0));

  }
}

void ImgStab::reset(TransformMat &mats) {
  mats.trmat = Matx33f (1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
  mats.trinv = Matx33f (1, 0, 0,
                   0, 1, 0,
                   0, 0, 1);
  // tracks.clear();
  // tracks_hist.clear();
  for (int i = 0; i < tracks.size(); i++) {
    tracks_hist[i].prev = tracks[i];
  }
  frame_idx = 0;
  if (!prev_frame.empty()) {
    find_points(prev_frame);
    frame_idx++;
  }
}

void ImgStab::compute_optflow(Mat const& frame) {
  if (!tracks.size()) return;
  out_tracks.clear();
  inv_tracks.clear();
  status.clear();
  status2.clear();
  // printf("init size: %d %d %d\n", out_tracks.size(), status.size(), inv_tracks.size());


  vector<float> err;
  Size winSize  = Size(15, 15);
  int maxLevel = 2;
  TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 10, 0.03);

  calcOpticalFlowPyrLK(prev_frame, frame, tracks, out_tracks, status, err, winSize, maxLevel, criteria);
  calcOpticalFlowPyrLK(frame, prev_frame, out_tracks, inv_tracks, status2, err, winSize, maxLevel, criteria);

  int next = 0;
  for (int i = 0; i < tracks.size(); i++) {
    float error = norm(tracks[i] - inv_tracks[i]);
    out_tracks[i + next] = out_tracks[i];
    tracks_hist[i + next] = tracks_hist[i];
    if (!status[i] || !status2[i] || (error > 1.0f)) {
      next--;
    }
  }
  out_tracks.resize(out_tracks.size() + next);
  tracks_hist.resize(tracks_hist.size() + next);

  for (auto track : out_tracks)
    circle(const_cast<Mat&>(frame), track, 5, 0, -1);


  swap(tracks, out_tracks);
  // printf("size: %d %d %d\n", out_tracks.size(), status.size(), inv_tracks.size());
}

void ImgStab::find_points(cv::Mat const& frame) {
    int maxCorners = 250;
    double qualityLevel = 0.05;
    double minDistance = 20;
    int blockSize = 7;

    mask = Mat::ones(frame.rows, frame.cols, CV_8UC1);
    for (auto &track : tracks) {
        circle(mask, track, minDistance, 0, -1);
    }

    vector<Point_<float> > new_tracks;
    goodFeaturesToTrack(frame, new_tracks, maxCorners, qualityLevel, minDistance, mask, blockSize);
    // tracks.insert(tracks.end(), new_tracks.begin(), new_tracks.end());
    for (auto &track : new_tracks) {
      if (track.y < frame.rows) {
        tracks.push_back(track);
        tracks_hist.push_back({track, track, 0});
      }
    }

}

#include <opencv2/opencv.hpp>
#include "image_stab.h"

#include "stdanalysis.h"
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;


static int threshold_value = 1;
// static int threshold_type = 1;
// static int threshold_sigma = 35;
int const max_value = 10000000;
// int const max_type = 4;


const char* trackbar_value = "Value";
// const char* trackbar_sigma = "Sigma";

deque<Mat> mats;

static void init() {
  /// Create a window to display results
  namedWindow( "Analysis", CV_WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  // createTrackbar( trackbar_type,
  //                 "Analysis", &threshold_type,
  //                 max_type);

  // createTrackbar( trackbar_sigma,
  //                 "Analysis", &threshold_sigma,
  //                 100);

  createTrackbar( trackbar_value,
                  "Analysis", &threshold_value,
                  max_value/10000);
}

Mat compute(Mat frame, Mat sq, Size size) {
  // float sigma = threshold_sigma/10.0f + 10e-5;

  blur(sq, sq, size);
  blur(frame, frame, size);
  cv::multiply(frame, frame, frame);

  double m, M;
  minMaxLoc(sq-frame, &m, &M);
  // printf("Min: %lf Max: %lf\n", m, M);

  return (sq-frame);
}

TransformMat gstabmat;

Mat analyze(cv::Mat frame, bool reset, ImgStab &stab) {
  srand(3213);

  if (reset) {
    mats.clear();
    stab.reset(gstabmat);
  }

  static bool isinit = false;
  if (!isinit) {
    init();
    isinit = true;
  }

  frame /= 256.0;

  cv::Mat sq;
  cv::multiply(frame, frame, sq);

  Mat stdx = compute(frame+0.0f, sq+0.0f, Size(6, 3));
  Mat stdy = compute(frame+0.0f, sq+0.0f, Size(3, 6));

  multiply(stdx, stdx, stdx);
  multiply(stdy, stdy, stdy);
  // GaussianBlur(sq, sq, size, sigma);
  // GaussianBlur(frame, frame, size, sigma);
  Mat img = min(stdx, stdy);
  // sqrt(img, img);

  stab.mat_update(gstabmat);

  // img = 1.0f-img;
  // stab.ftransform(gstabmat, img, true, 0, false);
  // img = 1.0f-img;



  threshold(img, img, (float)threshold_value/max_value, 1.0f, CV_THRESH_BINARY_INV);


  mats.push_back(img);
  while (mats.size() > 1) {
    mats.pop_front();
  }

  img.release();
  img = Mat::ones(mats.back().size(), CV_32F);

  for (auto m : mats) {
    cv::multiply(img, m, img);
  }


  // cout << "Threshold: " << ((float)threshold_value/max_value) << endl;

  int ersize = 2;
  Mat element1 = getStructuringElement( MORPH_ELLIPSE,
                                       Size( ersize*2+1, ersize*2+1 ),
                                       Point( ersize, ersize ));


  // erode(img, img, element1, Point(-1, -1), 1);
  for (int i = 0; i < 2; i++) {
    erode(img, img, element1, Point(-1, -1), 2);
    dilate(img, img, element1, Point(-1, -1), 2);
  }
  dilate(img, img, element1, Point(-1, -1), 1);
  erode(img, img, element1, Point(-1, -1), 1);
  // dilate(img, img, element1, Point(-1, -1), 3);


  // int erosion_size = 1;
  // Mat element = getStructuringElement( MORPH_RECT,
  //                                      Size( 2*erosion_size + 1, 2*erosion_size+1 ),
  //                                      Point( erosion_size, erosion_size ));
  // erode(img, img, element, Point(-1, -1), 3);


  // img *= 255.0f;
  // img.convertTo(img, CV_8U);
  return img;
//  imshow("TEST", img);

  // Create the marker image for the watershed algorithm
  Mat labels;
  Mat stats;
  Mat ctrs;
  int lc = connectedComponentsWithStats(img, labels, stats, ctrs, 4);

  // Generate random colors
  vector<Vec3b> colors;
  vector<pair<int, int> > areas;
  for (size_t i = 0; i < lc; i++)
  {
      int b = 255;//rand() % 255;
      int g = 255;//rand() % 255;
      int r = 255;//rand() % 255;
      colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
      if (i)
        areas.push_back(make_pair(stats.at<int>(i, CC_STAT_AREA), i-1));
  }
  sort(areas.rbegin(), areas.rend());
  // cout << string( 60, '\n' );
  vector<int> remaps; remaps.resize(areas.size());
  vector<int> testing; testing.resize(areas.size());

  for (int i = 0; i < areas.size(); i++) {
    remaps[areas[i].second] = i;
    testing[i] = 0;
  }
  // Create the result image
  Mat dst = Mat::zeros(img.size(), CV_8UC3);


  int totest = 1;
  // Fill labeled objects with random colors
  int lab = -1;
  for (int i = 0; i < img.rows; i++)
  {
      for (int j = 0; j < img.cols; j++)
      {
          int index = labels.at<int>(i,j);
          if (index > 0 && index <= static_cast<int>(lc)) {
              dst.at<Vec3b>(i,j) = colors[remaps[index-1]];
              lab = index-1;
              testing[remaps[index-1]]++;
          }
          else
              dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
      }
  }
  // cout << "Testing: " << testing[totest] << " col: " << colors[totest] << " exp: " << areas[totest].first <<
  //         " index: " << lab << " exp: " << areas[totest].second << " real: " << stats.at<int>(lab+1, CC_STAT_AREA) << endl;
  // assert(lab < 0 || testing[totest] == stats.at<int>(lab+1, CC_STAT_AREA));



  // imshow("Analysis", dst);
  // dst.copyTo(frame);
  // stab.ftransform(gstabmat, dst, true, 0, false, Point(0, 0), true);

  return dst;
}

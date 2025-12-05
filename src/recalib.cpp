/*
 * Copyright (c) 2022-2025, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "recalib.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

bool FisheyeSolvePnP(cv::InputArray opoints, cv::InputArray ipoints,
                     cv::InputArray cameraMatrix, cv::InputArray distCoeffs,
                     cv::OutputArray rvec, cv::OutputArray tvec) {

  cv::Mat imagePointsNormalized;
  cv::fisheye::undistortPoints(ipoints, imagePointsNormalized, cameraMatrix,
                               distCoeffs, cv::noArray());
  return cv::solvePnP(opoints, imagePointsNormalized, cv::Matx33d::eye(),
                      cv::noArray(), rvec, tvec);
}

cv::Matx33d ypr2R(const cv::Vec3d& ypr) {
  auto y = ypr(0) / 180.0 * std::numbers::pi;
  auto p = ypr(1) / 180.0 * std::numbers::pi;
  auto r = ypr(2) / 180.0 * std::numbers::pi;

  cv::Matx33d Rz(cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1);

  cv::Matx33d Ry(cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p));

  cv::Matx33d Rx(1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r));

  return Rz * Ry * Rx;
}

cv::Matx33d ypr2R(double y, double p, double r) {
  return ypr2R(cv::Vec3d(y, p, r));
}

cv::Vec3d R2ypr(const cv::Matx33d& R) {
  cv::Vec3d n(R(0, 0), R(1, 0), R(2, 0));
  cv::Vec3d o(R(0, 1), R(1, 1), R(2, 1));
  cv::Vec3d a(R(0, 2), R(1, 2), R(2, 2));

  cv::Vec3d ypr;
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r =
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));

  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / std::numbers::pi * 180.0;
}

bool findCornersForcedTopLeft(const cv::Mat& image, cv::Size patternSize,
                              std::vector<cv::Point2f>& corners) {
  bool found = cv::findChessboardCorners(image, patternSize, corners,
                                         cv::CALIB_CB_ADAPTIVE_THRESH |
                                             cv::CALIB_CB_NORMALIZE_IMAGE |
                                             cv::CALIB_CB_FAST_CHECK);

  if (found && !corners.empty()) {
    cv::Point2f p0 = corners.front();
    cv::Point2f pn = corners.back();

    float dist0 = p0.x * p0.x + p0.y * p0.y;
    float distn = pn.x * pn.x + pn.y * pn.y;

    if (dist0 > distn) {
      std::reverse(corners.begin(), corners.end());
    }
  }

  return found;
}

cv::Mat ChessboardCalibrator::i420_to_rgb(const std::string& yuv_path, int h,
                                          int w) {
  std::ifstream file(yuv_path, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Cannot open YUV file at: " << yuv_path << std::endl;
    return cv::Mat();
  }

  std::vector<uint8_t> yuv_data((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
  if (yuv_data.size() != static_cast<size_t>(w * h * 3 / 2)) {
    std::cerr << "Error: File size does not match expected YUV420 (I420) size."
              << std::endl;
    return cv::Mat();
  }

  cv::Mat rgb;
  cv::Mat I420(h * 3 / 2, w, CV_8UC1, yuv_data.data());
  cv::cvtColor(I420, rgb, cv::COLOR_YUV2RGB_I420);
  rgb_ = rgb.clone();

  return rgb;
}

ChessboardCalibrator::CalibResult
ChessboardCalibrator::chessboard_detect(const cv::Mat& rgb,
                                        const cv::Size& pattern_size,
                                        float square_size, bool is_fisheye) {
  ChessboardCalibrator::CalibResult res;

  cv::Mat gray;
  cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Point3f> objp;
  for (int i = 0; i < pattern_size.height; ++i) {
    for (int j = 0; j < pattern_size.width; ++j) {
      objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }

  std::vector<cv::Point2f> corners;

  // bool ret = cv::findChessboardCorners(gray, pattern_size, corners,
  //                                      cv::CALIB_CB_ADAPTIVE_THRESH |
  //                                          cv::CALIB_CB_NORMALIZE_IMAGE |
  //                                          cv::CALIB_CB_FAST_CHECK);
  bool ret = findCornersForcedTopLeft(gray, pattern_size, corners);
  if (!ret) {
    std::cerr << "Chessboard detection failed." << std::endl;
    return res;
  }

  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                            30, 0.001);
  cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

  cv::Mat rvec, tvec;
  bool pnp_success = false;
  if (is_fisheye) {
    // take the first 4 dist coeffs in dist_ for fisheye model
    cv::Vec<double, 4> fisheye_dist(dist_[0], dist_[1], dist_[2], dist_[3]);
    pnp_success = FisheyeSolvePnP(objp, corners, K_, fisheye_dist, rvec, tvec);
  } else {
    pnp_success = cv::solvePnP(objp, corners, K_, cv::Mat(dist_), rvec, tvec);
  }
  if (!pnp_success) {
    std::cerr << "SolvePnP failed." << std::endl;
  }

  cv::Matx33d R;
  cv::Rodrigues(rvec, R);

  auto ypr = R2ypr(R);

  res.angle_degrees = ypr;
  res.success = true;
  return res;
}

ChessboardCalibrator::CalibResult
ChessboardCalibrator::adaptive_chessboard_detect(const cv::Mat& rgb,
                                                 const cv::Vec4i& region,
                                                 const cv::Size& pattern_size,
                                                 float square_size,
                                                 bool is_fisheye) {
  ChessboardCalibrator::CalibResult res;

  cv::Mat gray;
  cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
  std::vector<cv::Point3f> objp;
  for (int i = 0; i < pattern_size.height; ++i) {
    for (int j = 0; j < pattern_size.width; ++j) {
      objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }

  int iter = 0;
  cv::Vec4i curr_region = region;
  cv::Mat valid_gray =
      gray(cv::Rect(curr_region[0], curr_region[1],
                    std::max(0, curr_region[2] - curr_region[0]),
                    std::max(0, curr_region[3] - curr_region[1])));

  std::vector<cv::Point2f> corners;
  bool ret = false;
  cv::Vec4i prev_region;

  while (!ret && iter < max_iters_) {
    // bool ret_ = cv::findChessboardCorners(valid_gray, pattern_size, corners,
    //                                       cv::CALIB_CB_ADAPTIVE_THRESH |
    //                                           cv::CALIB_CB_NORMALIZE_IMAGE |
    //                                           cv::CALIB_CB_FAST_CHECK);
    bool ret_ = findCornersForcedTopLeft(valid_gray, pattern_size, corners);
    if (ret_) {
      ret = ret_;
      break;
    }
    prev_region = curr_region;
    margin_marching(curr_region, stride_);
    if (curr_region == prev_region)
      break;

    if (curr_region[0] >= curr_region[2] || curr_region[1] >= curr_region[3])
      break;
    valid_gray = gray(cv::Rect(curr_region[0], curr_region[1],
                               curr_region[2] - curr_region[0],
                               curr_region[3] - curr_region[1]));
    iter++;
  }

  if (!ret) {
    std::cerr << "Chessboard detection failed." << std::endl;
    return res;
  }

  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                            30, 0.001);
  // Refine on ROI then offset to full-image coordinates
  cv::cornerSubPix(valid_gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                   criteria);
  for (auto& p : corners) {
    p.x += static_cast<float>(curr_region[0]);
    p.y += static_cast<float>(curr_region[1]);
  }

  cv::Mat rvec, tvec;
  bool pnp_success = false;
  if (is_fisheye) {
    // take the first 4 dist coeffs in dist_ for fisheye model
    cv::Vec<double, 4> fisheye_dist(dist_[0], dist_[1], dist_[2], dist_[3]);
    pnp_success = FisheyeSolvePnP(objp, corners, K_, fisheye_dist, rvec, tvec);
  } else {
    pnp_success = cv::solvePnP(objp, corners, K_, cv::Mat(dist_), rvec, tvec);
  }
  if (!pnp_success) {
    std::cerr << "SolvePnP failed." << std::endl;
  }
  cv::Matx33d R;
  cv::Rodrigues(rvec, R);

  auto ypr = R2ypr(R);

  res.angle_degrees = ypr;
  res.success = true;
  return res;
}

cv::Mat ChessboardCalibrator::get_warped_image(
    const ChessboardCalibrator::CalibResult& calib_res) const {
  auto ypr = calib_res.angle_degrees;
  auto h = rgb_.rows;
  auto w = rgb_.cols;
  cv::Mat warped;
  auto R = ypr2R(ypr);
  auto H = K_ * R.inv() * K_.inv();
  cv::warpPerspective(rgb_, warped, H, cv::Size(w, h));
  return warped;
}

cv::Mat ChessboardCalibrator::get_rgb_image() const { return rgb_; }

cv::Mat ChessboardCalibrator::im_read(const std::string& file_path) {
  cv::Mat rgb;
  if (file_path.ends_with(".yuv")) {
    std::cerr << "Error: For YUV files, please provide height and width."
              << std::endl;
    return cv::Mat();
  } else {
    rgb = cv::imread(file_path, cv::IMREAD_ANYCOLOR);
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  }
  rgb_ = rgb.clone();
  return rgb;
}

cv::Mat ChessboardCalibrator::im_read(const std::string& file_path, int h,
                                      int w) {
  cv::Mat rgb;
  if (file_path.ends_with(".yuv")) {
    rgb = i420_to_rgb(file_path, h, w);
  } else {
    rgb = cv::imread(file_path, cv::IMREAD_ANYCOLOR);
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  }
  rgb_ = rgb.clone();
  return rgb;
}

ChessboardCalibrator::CalibResult
ChessboardCalibrator::detect(const cv::Mat& rgb, const cv::Size& pattern_size,
                             float square_size, bool is_fisheye) {
  return chessboard_detect(rgb, pattern_size, square_size, is_fisheye);
}

ChessboardCalibrator::CalibResult ChessboardCalibrator::detect(
    const cv::Mat& rgb, const std::vector<std::vector<cv::Point2f>>& aruco_rois,
    const cv::Size& pattern_size, float square_size, bool is_fisheye) {
  auto region = identify_region(aruco_rois);
  return adaptive_chessboard_detect(rgb, region, pattern_size, square_size,
                                    is_fisheye);
}

cv::Mat ChessboardCalibrator::mask_aruco(
    const cv::Mat& rgb, const std::vector<std::vector<cv::Point2f>>& rois) {
  cv::Mat mask = cv::Mat::zeros(rgb.size(), rgb.type());

  auto roi = identify_region(rois);
  int x_min_int = roi[0];
  int y_min_int = roi[1];
  int x_max_int = roi[2];
  int y_max_int = roi[3];

  rgb(cv::Rect(x_min_int, y_min_int, x_max_int - x_min_int,
               y_max_int - y_min_int))
      .copyTo(mask(cv::Rect(x_min_int, y_min_int, x_max_int - x_min_int,
                            y_max_int - y_min_int)));

  return mask;
}

cv::Vec4i ChessboardCalibrator::identify_region(
    const std::vector<std::vector<cv::Point2f>>& rois) {
  std::vector<cv::Point2f> centers;
  for (const auto& roi : rois) {
    cv::Point2f center(0, 0);
    for (const auto& pt : roi) {
      center.x += pt.x;
      center.y += pt.y;
    }
    center.x /= roi.size();
    center.y /= roi.size();
    centers.push_back(center);
  }

  int left_idx = 0, right_idx = 0, top_idx = 0, bottom_idx = 0;
  for (size_t i = 0; i < centers.size(); ++i) {
    if (centers[i].x < centers[left_idx].x)
      left_idx = i;
    if (centers[i].x > centers[right_idx].x)
      right_idx = i;
    if (centers[i].y < centers[top_idx].y)
      top_idx = i;
    if (centers[i].y > centers[bottom_idx].y)
      bottom_idx = i;
  }

  const auto& left_roi = rois[left_idx];
  float rightmost_x = 0;
  for (const auto& pt : left_roi) {
    rightmost_x = std::max(rightmost_x, pt.x);
  }
  cv::Point2f left_corner(rightmost_x, centers[left_idx].y);

  const auto& right_roi = rois[right_idx];
  float leftmost_x = std::numeric_limits<float>::max();
  for (const auto& pt : right_roi) {
    leftmost_x = std::min(leftmost_x, pt.x);
  }
  cv::Point2f right_corner(leftmost_x, centers[right_idx].y);

  const auto& top_roi = rois[top_idx];
  float bottommost_y = 0;
  for (const auto& pt : top_roi) {
    bottommost_y = std::max(bottommost_y, pt.y);
  }
  cv::Point2f top_corner(centers[top_idx].x, bottommost_y);

  const auto& bottom_roi = rois[bottom_idx];
  float topmost_y = std::numeric_limits<float>::max();
  for (const auto& pt : bottom_roi) {
    topmost_y = std::min(topmost_y, pt.y);
  }
  cv::Point2f bottom_corner(centers[bottom_idx].x, topmost_y);

  std::vector<cv::Point2f> corners = {left_corner, right_corner, top_corner,
                                      bottom_corner};
  float x_min = corners[0].x, y_min = corners[0].y;
  float x_max = corners[0].x, y_max = corners[0].y;
  for (const auto& pt : corners) {
    x_min = std::min(x_min, pt.x);
    y_min = std::min(y_min, pt.y);
    x_max = std::max(x_max, pt.x);
    y_max = std::max(y_max, pt.y);
  }

  int x_min_int = static_cast<int>(x_min);
  int y_min_int = static_cast<int>(y_min);
  int x_max_int = static_cast<int>(x_max);
  int y_max_int = static_cast<int>(y_max);

  return cv::Vec4i(x_min_int, y_min_int, x_max_int, y_max_int);
}

void ChessboardCalibrator::margin_marching(cv::Vec4i& region, int stride) {
  region[0] = region[0] + stride;
  region[1] = region[1] + stride;
  region[2] = std::max(0, region[2] - stride);
  region[3] = std::max(0, region[3] - stride);
}
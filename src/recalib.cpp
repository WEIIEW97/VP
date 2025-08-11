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

#include <fstream>
#include <iostream>
#include <vector>

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

ChessboardCalibrator::CalibResult ChessboardCalibrator::chessboard_detect(
    const cv::Mat& rgb, const cv::Size& pattern_size, float square_size) {
  ChessboardCalibrator::CalibResult res;

  cv::Mat gray;
  cv::cvtColor(rgb_, gray, cv::COLOR_RGB2GRAY);
  std::vector<cv::Point3f> objp;
  for (int i = 0; i < pattern_size.height; ++i) {
    for (int j = 0; j < pattern_size.width; ++j) {
      objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }

  std::vector<cv::Point2f> corners;
  bool ret = cv::findChessboardCorners(gray, pattern_size, corners,
                                       cv::CALIB_CB_ADAPTIVE_THRESH |
                                           cv::CALIB_CB_NORMALIZE_IMAGE);
  if (!ret) {
    std::cerr << "Chessboard detection failed." << std::endl;
    return res;
  }

  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                            30, 0.001);
  cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

  cv::Mat rvec, tvec;
  bool pnp_success =
      cv::solvePnP(objp, corners, K_, cv::Mat(dist_), rvec, tvec);
  if (!pnp_success) {
    std::cerr << "SolvePnP failed." << std::endl;
  }

  cv::Mat R;
  cv::Rodrigues(rvec, R);

  double pitch = std::atan2(-R.at<double>(2, 0),
                            std::sqrt(std::pow(R.at<double>(2, 1), 2) +
                                      std::pow(R.at<double>(2, 2), 2))) *
                 180.0 / CV_PI;
  double yaw =
      std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0 / CV_PI;
  double roll =
      std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)) * 180.0 / CV_PI;
  res.angle_degrees = cv::Vec3d(pitch, yaw, roll);
  res.success = true;
  return res;
}

cv::Mat ChessboardCalibrator::get_warped_image(
    const ChessboardCalibrator::CalibResult& calib_res) const {
  auto yaw = calib_res.angle_degrees[1];
  auto h = rgb_.rows;
  auto w = rgb_.cols;
  auto opticial_center = cv::Point2f(K_.at<double>(0, 2), K_.at<double>(1, 2));

  auto rmat = cv::getRotationMatrix2D(opticial_center, yaw, 1);
  cv::Mat warped;
  cv::warpAffine(rgb_, warped, rmat, rgb_.size());
  return warped;
}

cv::Mat ChessboardCalibrator::get_rgb_image() const { return rgb_; }

ChessboardCalibrator::CalibResult
ChessboardCalibrator::detect(const std::string& yuv_path, int h, int w,
                             const cv::Size& pattern_size, float square_size) {
  auto rgb = i420_to_rgb(yuv_path, h, w);
  return chessboard_detect(rgb, pattern_size, square_size);
}
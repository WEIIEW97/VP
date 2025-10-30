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

#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>

class ChessboardCalibrator {
public:
  ChessboardCalibrator(const cv::Matx33d& K, const cv::Vec<double, 8>& dist)
      : K_(K), dist_(dist) {}
  ChessboardCalibrator(const cv::Matx33d& K, const cv::Vec<double, 8>& dist,
                       int stride)
      : K_(K), dist_(dist), stride_(stride) {}
  ChessboardCalibrator(const cv::Matx33d& K, const cv::Vec<double, 8>& dist,
                       int stride, int max_iters)
      : K_(K), dist_(dist), stride_(stride), max_iters_(max_iters) {}

  ~ChessboardCalibrator() = default;

  struct CalibResult {
    bool success = false;
    cv::Vec3d angle_degrees;
    std::vector<cv::Point2f> corners;
    cv::Size pattern_size;
  };

  cv::Mat im_read(const std::string& file_path);
  cv::Mat im_read(const std::string& file_path, int h, int w);

  CalibResult detect(const cv::Mat& rgb,
                     const cv::Size& pattern_size = cv::Size(8, 5),
                     float square_size = 0.025, bool is_fisheye = false);
  CalibResult detect(const cv::Mat& rgb,
                     const std::vector<std::vector<cv::Point2f>>& aruco_rois,
                     const cv::Size& pattern_size = cv::Size(8, 5),
                     float square_size = 0.025, bool is_fisheye = false);

  cv::Mat get_warped_image(const CalibResult& calib_res) const;
  cv::Mat get_rgb_image() const;
  void set_stride(int stride) { stride_ = stride; }
  void set_max_iters(int max_iters) { max_iters_ = max_iters; }

private:
  cv::Mat i420_to_rgb(const std::string& yuv_path, int h, int w);
  CalibResult chessboard_detect(const cv::Mat& rgb,
                                const cv::Size& pattern_size, float square_size,
                                bool is_fisheye);
  CalibResult adaptive_chessboard_detect(const cv::Mat& rgb, const cv::Vec4i& region,
                                         const cv::Size& pattern_size,
                                         float square_size, bool is_fisheye);
  cv::Mat mask_aruco(const cv::Mat& rgb,
                     const std::vector<std::vector<cv::Point2f>>& rois);

  cv::Vec4i identify_region(const std::vector<std::vector<cv::Point2f>>& rois);
  void margin_marching(cv::Vec4i& region, int stride = 4);

private:
  cv::Mat rgb_;
  cv::Mat K_;
  cv::Vec<double, 8> dist_;
  int stride_ = 4;
  int max_iters_ = 20;
};

cv::Matx33d ypr2R(double pitch, double yaw, double roll);
cv::Matx33d ypr2R(const cv::Vec3d& ypr);

cv::Vec3d R2ypr(const cv::Matx33d& R);
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
  ~ChessboardCalibrator() = default;

  struct CalibResult {
    bool success = false;
    cv::Vec3d angle_degrees;
    std::vector<cv::Point2f> corners;
    cv::Size pattern_size;
  };

  CalibResult detect(const std::string& file_path, int h, int w,
                     const cv::Size& pattern_size = cv::Size(8, 5),
                     float square_size = 0.025);

  cv::Mat get_warped_image(const CalibResult& calib_res) const;
  cv::Mat get_rgb_image() const;

private:
  cv::Mat i420_to_rgb(const std::string& yuv_path, int h, int w);
  CalibResult chessboard_detect(const cv::Mat& rgb,
                                const cv::Size& pattern_size,
                                float square_size);

private:
  cv::Mat rgb_;
  cv::Mat K_;
  cv::Vec<double, 8> dist_;
};

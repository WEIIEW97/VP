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

// CamCalibTool - A camera calibration tool
// Copyright (c) 2022, Algorithm Development Team of NextVPU (Shanghai) Co.,
// Ltd. All rights reserved.
//
// This software was developed of Jacob.lsx
//
// Use AprilTag-Marker detect corners

#pragma once

#include <map>
#include <opencv2/core/core.hpp>

#include "apriltags/TagDetector.h"
#include "apriltags/Tag36h11.h"

class Aprilgrid {
public:
  typedef struct {
    cv::Size board_size;
    float marker_length;
    float marker_separation;
  } AprilgridConfig;

  typedef struct {
    cv::Mat rvec;
    cv::Mat tvec;
  } Pose;

  Aprilgrid(AprilgridConfig& config, cv::Mat& image);

  std::map<int, int> findCorners(int minimum_valid_april_threshold = 5);
  const std::map<int, std::vector<std::pair<cv::Point2f, cv::Point3f>>>&
  getCorners(void) const;
  bool cornersFound(void) const;

  const cv::Mat& getImage(void) const;
  const cv::Mat& getSketch(void) const;

  Pose estimatePose(
      const std::map<int, std::vector<std::pair<cv::Point2f, cv::Point3f>>>&
          detections,
      const cv::Mat& K, const cv::Vec<double, 8>& dist_coeff, int id_end = -1);

private:
  void draw(const std::vector<AprilTags::TagDetection>& tag_detection);

  /// \brief max. displacement squarred in subpixel refinement  [px^2]
  double max_subpix_displacement2_;
  /// \brief min. distance form image border for valid points [px]
  double min_border_distance_;
  /// \brief size of black border around the tag code bits (in pixels)
  unsigned int black_tag_border_;

  cv::Mat image_;
  cv::Mat sketch_;
  std::map<int, std::vector<std::pair<cv::Point2f, cv::Point3f>>> corners_;
  AprilgridConfig board_config_;
  bool corners_found_;
};

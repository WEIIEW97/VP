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

#include "aprilgrid2.h"
#include <set>
#include <opencv2/imgproc.hpp>
#include <memory>

Aprilgrid::Aprilgrid(AprilgridConfig& config)
    : max_subpix_displacement2_(5.991), min_border_distance_(4.0),
      black_tag_border_(2), board_config_(config), corners_found_(false) {}

void Aprilgrid::feed(const cv::Mat& image) {
  CV_Assert(!image.empty());
  if (image.channels() == 1) {
    image.copyTo(image_);
  } else {
    cv::cvtColor(image, image_, cv::COLOR_BGR2GRAY);
  }
  image_.copyTo(sketch_);
}

std::map<int, int> Aprilgrid::findCorners(int minimum_valid_april_threshold) {
  // create the tag detector
  std::shared_ptr<AprilTags::TagDetector> tag_detector =
      std::make_shared<AprilTags::TagDetector>(AprilTags::tagCodes36h11,
                                               black_tag_border_);

  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;

  // detect the tags
  std::vector<AprilTags::TagDetection> detections =
      tag_detector->extractTags(image_);
  std::map<int, int> num_marker_in_board; // < board_id, marker_num >

  // min. distance [px] of tag corners from image border (tag is not used if
  // violated)
  std::vector<AprilTags::TagDetection>::iterator iter = detections.begin();
  for (iter = detections.begin(); iter != detections.end();) {
    // check all four corners for violation
    bool remove = false;

    for (int j = 0; j < 4; j++) {
      remove |= iter->p[j].first < min_border_distance_;
      remove |= iter->p[j].first >
                (float)(image_.cols) - min_border_distance_; // width
      remove |= iter->p[j].second < min_border_distance_;
      remove |= iter->p[j].second >
                (float)(image_.rows) - min_border_distance_; // height
    }

    // also remove tags that are flagged as bad
    if (iter->good != true)
      remove |= true;

    /// supported multiple boards
    //    // also remove if the tag ID is out-of-range for this grid (faulty
    //    detection) if (iter->id >= (int)(board_config_.board_size.width *
    //    board_config_.board_size.height))
    //      remove |= true;

    // delete flagged tags
    if (remove) {
      // delete the tag and advance in list
      iter = detections.erase(iter);
    } else {
      // advance in list
      ++iter;
    }
  }

  if (detections.empty())
    return num_marker_in_board;
  else
    corners_found_ = true;

  const int add_x[4] = {0, 1, 1, 0};
  const int add_y[4] = {0, 0, 1, 1};
  const int marker_number_of_one_board =
      board_config_.board_size.width * board_config_.board_size.height;
  std::set<int> marker_id_set;
  for (int i = 0; i < detections.size(); i++) {
    int id = detections.at(i).id;
    int board_id = id / marker_number_of_one_board;
    if (num_marker_in_board.count(board_id)) {
      ++num_marker_in_board.at(board_id);
    } else {
      num_marker_in_board[board_id] = 1;
    }

    // If the same marker ID is detected, it is discarded directly
    if (marker_id_set.count(id)) {
      corners_.erase(id);
      continue;
    } else
      marker_id_set.insert(id);

    for (int j = 0; j < 4; j++) {
      float u = detections.at(i).p[j].first;
      float v = detections.at(i).p[j].second;

      int row =
          id % marker_number_of_one_board / board_config_.board_size.width;
      int col =
          id % marker_number_of_one_board % board_config_.board_size.width;
      auto point = cv::Point3f(col * (board_config_.marker_length +
                                      board_config_.marker_separation) +
                                   add_x[j] * board_config_.marker_length,
                               row * (board_config_.marker_length +
                                      board_config_.marker_separation) +
                                   add_y[j] * board_config_.marker_length,
                               0.f);
      corners_[id].emplace_back(std::make_pair(cv::Point2f(u, v), point));
    }
  }

  // optional subpixel refinement on all tag corners (four corners each tag)
  std::vector<cv::Point2f> image_points_rf;
  for (auto& it_quad : corners_) {
    for (auto& it : it_quad.second) {
      image_points_rf.push_back(it.first);
    }
  }
  cv::cornerSubPix(image_, image_points_rf, cv::Size(5, 5), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::Type::EPS +
                                        cv::TermCriteria::Type::MAX_ITER,
                                    30, 0.1));

  // remove outlier marker that corners if the displacement in the subpixel
  // refinement is bigger a given threshold
  {
    int i = 0;
    for (auto it_quad = corners_.begin(); it_quad != corners_.end(); ++i) {
      bool remove = false;
      int j = 0;
      for (auto& it : it_quad->second) {
        // refined corners
        const auto& corner = image_points_rf.at(4 * i + j);
        // raw corners
        const auto& corner_raw = it.first;

        double subpix_displacement_squarred = cv::norm(corner - corner_raw);
        subpix_displacement_squarred *= subpix_displacement_squarred;

        if (subpix_displacement_squarred > max_subpix_displacement2_) {
          remove = true;
          break;
        }

        ++j;
      }
      if (remove) {
        --num_marker_in_board.at(it_quad->first / marker_number_of_one_board);
        it_quad = corners_.erase(it_quad);
      } else {
        j = 0;
        for (auto& it : it_quad->second) {
          it.first = image_points_rf.at(4 * i + j);
          ++j;
        }
        ++it_quad;
      }
    }
  }

  for (auto iter = num_marker_in_board.begin();
       iter != num_marker_in_board.end();) {
    if (iter->second < minimum_valid_april_threshold) {
      for (int id = iter->first * marker_number_of_one_board;
           id < (iter->first + 1) * marker_number_of_one_board; ++id) {
        if (corners_.count(id))
          corners_.erase(id);
      }
      iter = num_marker_in_board.erase(iter);
    } else {
      ++iter;
    }
  }

  if (num_marker_in_board.empty()) {
    corners_found_ = false;
    return num_marker_in_board;
  }

  //// Debug
  // for (const auto& it : corners_) {
  //   std::printf("Id: %d, corner1: [%.4f %.4f], corner2: [%.4f %.4f],"
  //     "corner3: [%.4f %.4f], corner4: [%.4f %.4f] \r\n", it.first,
  //     it.second[0].first.x, it.second[0].first.y,
  //     it.second[1].first.x, it.second[1].first.y,
  //     it.second[2].first.x, it.second[2].first.y,
  //     it.second[3].first.x, it.second[3].first.y);
  // }
  // std::cout << std::endl;

  draw(detections);

  return num_marker_in_board;
}

const std::map<int, std::vector<std::pair<cv::Point2f, cv::Point3f>>>&
Aprilgrid::getCorners(void) const {
  return corners_;
}

bool Aprilgrid::cornersFound(void) const { return corners_found_; }

const cv::Mat& Aprilgrid::getImage(void) const { return image_; }

const cv::Mat& Aprilgrid::getSketch(void) const { return sketch_; }

void Aprilgrid::draw(
    const std::vector<AprilTags::TagDetection>& tag_detection) {
  for (auto& item : tag_detection) {
    int id = item.id;
    if (corners_.count(id)) {
      item.draw(sketch_);
      const auto& corner_quad = corners_.at(id);
      for (int i = 0; i < 4; ++i) {
        const auto& point = corner_quad.at(i).first;
        cv::circle(sketch_, point, 2, cv::Scalar(0, 255, 0, 0), 1);
      }
    }
  }
}

Aprilgrid::Pose Aprilgrid::estimatePose(
    const std::map<int, std::vector<std::pair<cv::Point2f, cv::Point3f>>>&
        detections,
    const cv::Mat& K, const cv::Vec<double, 8>& dist_coeff, int id_end) {
  Pose pose;
  if (detections.empty())
    return pose;

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  for (const auto& it : detections) {
    if (id_end != -1 && it.first >= id_end)
      continue;
    for (const auto& corner : it.second) {
      image_points.push_back(corner.first);
      object_points.push_back(corner.second);
    }
  }

  cv::solvePnP(object_points, image_points, K, dist_coeff, pose.rvec, pose.tvec,
               false, cv::SOLVEPNP_ITERATIVE);

  return pose;
}

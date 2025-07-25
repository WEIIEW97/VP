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

#include "stitch.h"

#include "ipm.h"
#include <opencv2/calib3d.hpp>
#include <fmt/format.h>

cv::Mat get_homogeneous_transform(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Matx33d R;
  cv::Rodrigues(rvec, R);

  cv::Mat T_top, T_bottom, T;
  cv::hconcat(R, tvec, T_top);

  T_bottom = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
  cv::vconcat(T_top, T_bottom, T);
  return T;
}

BEVPack process_single_bev(cv::Mat& im_rgb, Aprilgrid& detector,
                           const IPM::IPMInfo& ipm_info, const cv::Matx33d& K,
                           const cv::Vec<double, 8>& dist) {
  BEVPack bev_pack;

  cv::Mat im;
  cv::cvtColor(im_rgb, im, cv::COLOR_BGR2GRAY);
  auto map_id = detector.findCorners();
  auto corners = detector.getCorners();
  cv::Mat K_mat = cv::Mat(K);
  auto ret = detector.estimatePose(corners, K_mat, dist, 1);
  if (ret.rvec.empty() || ret.tvec.empty()) {
    fmt::print("Pose estimation failed, returning empty BEVPack.\n");
    return bev_pack; // Return empty BEVPack if pose estimation fails
  }

  auto T_c_b = get_homogeneous_transform(ret.rvec, ret.tvec);
  cv::Matx33d R_c_b;
  cv::Rodrigues(ret.rvec, R_c_b);
  auto ipm = IPM(K, dist, im.size(), R_c_b, ret.tvec);
  auto bev_image = ipm.GetIPMImage(im_rgb, ipm_info, R_c_b);
  bev_pack.bev_image = bev_image;
  bev_pack.T = T_c_b;

  return bev_pack;
};

cv::Mat stitch(const std::vector<std::string>& image_paths, float ipm_x_scale,
               float ipm_y_scale, int ref_image_idx, bool show_result) {
  auto ap_config = Aprilgrid::AprilgridConfig();
  ap_config.board_size = cv::Size(6, 4);
  ap_config.marker_length = 0.04;
  ap_config.marker_separation = 0.012;

  auto ap_detector = Aprilgrid(ap_config);

  IPM::IPMInfo ipm_info;
  ipm_info.x_scale = ipm_x_scale;
  ipm_info.y_scale = ipm_y_scale;

  cv::Matx33d K(700.0400, 0, 630.6500, 0, 700.0400, 331.5925, 0, 0, 1);
  cv::Vec<double, 8> dist(-0.1724, 0.0270, 0.0024, 0.0003, 0, 0, 0, 0);
  
  std::vector<cv::Mat> images;
  std::vector<BEVPack> packs;
  for (const auto& path : image_paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_ANYCOLOR);
    if (image.empty()) {
      fmt::print("Failed to read image: {}\n", path);
      continue;
    }
    images.push_back(image);
    ap_detector.feed(image);
    auto pack = process_single_bev(image, ap_detector, ipm_info, K, dist);
    packs.push_back(pack);
  }

  cv::Mat total_bev = cv::Mat::zeros(packs[0].bev_image.size(), CV_64FC3);
  
}

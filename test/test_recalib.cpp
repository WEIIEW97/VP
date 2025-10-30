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
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "../src/json.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper function to get directories starting with a prefix
std::vector<std::string> get_dirs(const std::string& root_dir,
                                  const std::string& prefix) {
  std::vector<std::string> dirs;
  for (const auto& entry : fs::directory_iterator(root_dir)) {
    if (entry.is_directory() &&
        entry.path().filename().string().starts_with(prefix)) {
      dirs.push_back(entry.path().string());
    }
  }
  return dirs;
}

std::vector<std::string> get_files(const std::string& dir,
                                   const std::string& ext) {
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ext) {
      files.push_back(entry.path().string());
    }
  }
  return files;
}

struct CameraParams {
  cv::Mat K;
  cv::Vec<double, 8> dist_coef = cv::Vec<double, 8>::all(0);
  bool is_fisheye;
};

CameraParams load_yaml(const std::string& yaml_path) {
  cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Cannot open YAML file: " + yaml_path);
  }

  std::string model_type;
  fs["model_type"] >> model_type;

  CameraParams params;

  if (model_type == "PINHOLE_FULL") {
    double fx, fy, cx, cy;
    fx = static_cast<double>(fs["projection_parameters"]["fx"]);
    fy = static_cast<double>(fs["projection_parameters"]["fy"]);
    cx = static_cast<double>(fs["projection_parameters"]["cx"]);
    cy = static_cast<double>(fs["projection_parameters"]["cy"]);

    params.K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    int idx = 0;
    for (const auto& key : {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"}) {
      params.dist_coef[idx++] =
          static_cast<double>(fs["distortion_parameters"][key]);
    }
    params.is_fisheye = false;
  } else if (model_type == "KANNALA_BRANDT") {
    double mu, mv, u0, v0;
    mu = static_cast<double>(fs["projection_parameters"]["mu"]);
    mv = static_cast<double>(fs["projection_parameters"]["mv"]);
    u0 = static_cast<double>(fs["projection_parameters"]["u0"]);
    v0 = static_cast<double>(fs["projection_parameters"]["v0"]);

    params.K = (cv::Mat_<double>(3, 3) << mu, 0, u0, 0, mv, v0, 0, 0, 1);

    int idx = 0;
    for (const auto& key : {"k2", "k3", "k4", "k5"}) {
      params.dist_coef[idx++] =
          static_cast<double>(fs["projection_parameters"][key]);
    }
    params.is_fisheye = true;
  } else {
    throw std::runtime_error("Unsupported model type: " + model_type);
  }
  fs.release();
  return params;
}

const std::vector<std::vector<cv::Point2f>> rois = {
  {{1400.041, 739.371}, {1401.901, 432.779}, {1787.924, 435.121}, {1786.064, 741.713}},
  {{732.485, 61.231}, {1196.261, 60.705}, {1196.500, 270.821}, {732.723, 271.347}},
  {{689.009, 868.880}, {1196.861, 867.400}, {1197.337, 1030.711}, {689.484, 1032.191}},
  {{107.398, 457.852}, {470.870, 447.373}, {478.764, 721.162}, {115.292, 731.641}}
};

const std::vector<std::vector<cv::Point2f>> rois_scaled = {
  {{1400.041 * 0.5, 739.371 * 0.5}, {1401.901 * 0.5, 432.779 * 0.5}, {1787.924 * 0.5, 435.121 * 0.5}, {1786.064 * 0.5, 741.713 * 0.5}},
  {{732.485 * 0.5, 61.231 * 0.5}, {1196.261 * 0.5, 60.705 * 0.5}, {1196.500 * 0.5, 270.821 * 0.5}, {732.723 * 0.5, 271.347 * 0.5}},
  {{689.009 * 0.5, 868.880 * 0.5}, {1196.861 * 0.5, 867.400 * 0.5}, {1197.337 * 0.5, 1030.711 * 0.5}, {689.484 * 0.5, 1032.191 * 0.5}},
  {{107.398 * 0.5, 457.852 * 0.5}, {470.870 * 0.5, 447.373 * 0.5}, {478.764 * 0.5, 721.162 * 0.5}, {115.292 * 0.5, 731.641 * 0.5}}
};

void test_patch() {
  std::string root_dir = "/home/william/extdisk/data/calib/failed/20251030/";
  // retrieve all directories in root_dir if begin with "abnor"
  std::vector<std::string> dirs = get_dirs(root_dir, "Z0CABLB25IRA");
  for (const auto& dir : dirs) {
    fs::path intri_path = fs::path(root_dir) /
                          fs::path(dir).filename() / "result" /
                          "RGB.yaml";
    fs::path img_path = fs::path(dir) / "RGB";
    std::vector<std::string> img_files = get_files(img_path.string(), ".png");
    auto params = load_yaml(intri_path.string());
    auto K = params.K;
    auto half_k = K.clone();
    // half_k(cv::Rect(0, 0, 2, 2)) = half_k(cv::Rect(0, 0, 2, 2)) * 0.5;
    half_k.at<double>(0, 0) *= 0.5;
    half_k.at<double>(1, 1) *= 0.5;
    half_k.at<double>(0, 2) *= 0.5;
    half_k.at<double>(1, 2) *= 0.5;

    auto dist = params.dist_coef;
    auto is_fisheye = params.is_fisheye;
    auto calibrator = ChessboardCalibrator(half_k, dist);
    for (const auto& img_file : img_files) {
      auto rgb = calibrator.im_read(img_file);
      cv::resize(rgb, rgb, rgb.size() / 2);
      auto res = calibrator.detect(rgb, rois_scaled,cv::Size(6, 3), 0.08, is_fisheye);
      std::cout << "file path is: " << img_file << std::endl;
      if (res.success) {
        std::cout << "Calibration result is: (yaw, pitch, roll) in degrees "
                  << res.angle_degrees << std::endl;
        auto verbose_img = calibrator.get_warped_image(res);
        // resize to 1/2
        // cv::resize(verbose_img, verbose_img, cv::Size(), 0.5, 0.5);
        cv::imshow("warped image", verbose_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
      } else {
        std::cout << "Failed to run calibration!" << std::endl;
      }
    }
  }
}

void test_single() {
  std::string img_path = "/home/william/extdisk/data/calib/image_save/abnor-63/"
                         "RGB/rgbvi-2025-8-22-12-2-16.png";
  std::string intri_path =
      "/home/william/extdisk/data/calib/image_save/calibration_intrinsic/"
      "abnor-63/result/intrinsics_colin.json";
  auto params = load_yaml(intri_path);
  auto K = params.K;
  auto dist = params.dist_coef;
  auto is_fisheye = params.is_fisheye;
  auto calibrator = ChessboardCalibrator(K, dist);
  auto rgb = calibrator.im_read(img_path);
  auto res = calibrator.detect(rgb, cv::Size(6, 3), 0.08, is_fisheye);
  if (res.success) {
    std::cout << "Calibration result is: (yaw, pitch, roll) in degrees "
              << res.angle_degrees << std::endl;
    auto verbose_img = calibrator.get_warped_image(res);
    cv::imshow("warped image", verbose_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
  } else {
    std::cout << "Failed to run calibration!" << std::endl;
  }
}

void test_aruco_mask() {
  std::string img_path = 
      "/home/william/extdisk/data/calib/failed/20251028/Z0CABLB25IRA0061#BA.04.00.0069.01-nodetection/RGB/rgbvi-2025-10-28-14-59-10-nodetection.png";
  std::string intri_path = 
      "/home/william/extdisk/data/calib/failed/20251028/Z0CABLB25IRA0061#BA.04.00.0069.01-nodetection/result/RGB.yaml";
  
  auto params = load_yaml(intri_path);
  auto K = params.K;
  auto dist = params.dist_coef;
  auto is_fisheye = params.is_fisheye;
  
  std::cout << "K is: " << K << std::endl;
  std::cout << "dist_coef is: " << dist << std::endl;
  std::cout << "flag_is_fisheye is: " << is_fisheye << std::endl;
  
  auto calibrator = ChessboardCalibrator(K, dist);
  auto rgb = calibrator.im_read(img_path);
  auto res = calibrator.detect(rgb, rois, cv::Size(6, 3), 0.08, is_fisheye);

  
  std::cout << "detect angle offset is: " << res.angle_degrees << std::endl;
  
  if (res.success) {
    if (cv::norm(res.angle_degrees[0]) >= 1.5 || 
        cv::norm(res.angle_degrees[1]) >= 1.5 || 
        cv::norm(res.angle_degrees[2]) >= 1.5) {
      std::cout << "Exceed the offset threshold of 1.5 degrees." << std::endl;
      return;
    }
    
    auto corrected_im = calibrator.get_warped_image(res);
    cv::imshow("corrected image", corrected_im);
    cv::waitKey(0);
    cv::destroyAllWindows();
  } else {
    std::cout << "Failed to detect chessboard in image." << std::endl;
  }
}

int main() {
  test_patch();
  return 0;
}
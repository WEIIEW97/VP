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

void test_patch() {
  std::string root_dir = "/home/william/extdisk/data/calib/failed/20251028/";
  // retrieve all directories in root_dir if begin with "abnor"
  std::vector<std::string> dirs = get_dirs(root_dir, "Z0CABLB25IRA0061");
  for (const auto& dir : dirs) {
    fs::path intri_path = fs::path(root_dir) /
                          fs::path(dir).filename() / "result" /
                          "RGB.yaml";
    fs::path img_path = fs::path(dir) / "RGB";
    std::vector<std::string> img_files = get_files(img_path.string(), ".png");
    auto params = load_yaml(intri_path.string());
    auto K = params.K;
    auto dist = params.dist_coef;
    auto is_fisheye = params.is_fisheye;
    auto calibrator = ChessboardCalibrator(K, dist);
    for (const auto& img_file : img_files) {
      auto res = calibrator.detect(img_file, 1080, 1920, cv::Size(6, 3), 0.08, is_fisheye);
      std::cout << "file path is: " << img_file << std::endl;
      if (res.success) {
        std::cout << "Calibration result is: (yaw, pitch, roll) in degrees "
                  << res.angle_degrees << std::endl;
        auto verbose_img = calibrator.get_warped_image(res);
        // resize to 1/2
        cv::resize(verbose_img, verbose_img, cv::Size(), 0.5, 0.5);
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
  auto res = calibrator.detect(img_path, 1080, 1920, cv::Size(6, 3), 0.08, is_fisheye);
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

int main() {
  test_patch();
  return 0;
}
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

void test_patch() {
  std::string root_dir = "/home/william/extdisk/data/calib/image_save";
  // retrieve all directories in root_dir if begin with "abnor"
  std::vector<std::string> dirs = get_dirs(root_dir, "abnor");
  for (const auto& dir : dirs) {
    fs::path intri_path = fs::path(root_dir) / "calibration_intrinsic" /
                          fs::path(dir).filename() / "result" /
                          "intrinsics_colin.json";
    fs::path img_path = fs::path(dir) / "RGB";
    std::vector<std::string> img_files = get_files(img_path.string(), ".png");
    auto intri = read_json(intri_path.string());
    cv::Matx33d K(intri["cam_intrinsic"][0], intri["cam_intrinsic"][1],
                  intri["cam_intrinsic"][2], intri["cam_intrinsic"][3],
                  intri["cam_intrinsic"][4], intri["cam_intrinsic"][5],
                  intri["cam_intrinsic"][6], intri["cam_intrinsic"][7],
                  intri["cam_intrinsic"][8]);
    cv::Vec<double, 8> dist(
        intri["cam_distcoeffs"][0], intri["cam_distcoeffs"][1],
        intri["cam_distcoeffs"][2], intri["cam_distcoeffs"][3],
        intri["cam_distcoeffs"][4], intri["cam_distcoeffs"][5],
        intri["cam_distcoeffs"][6], intri["cam_distcoeffs"][7]);
    auto calibrator = ChessboardCalibrator(K, dist);
    for (const auto& img_file : img_files) {
      auto res = calibrator.detect(img_file, 1080, 1920, cv::Size(6, 3));
      if (res.success) {
        std::cout << "Calibration result is: (pitch, yaw, roll) in degrees "
                  << res.angle_degrees << std::endl;
        auto verbose_img = calibrator.get_warped_image(res);
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
  auto intri = read_json(intri_path);
  cv::Matx33d K(intri["cam_intrinsic"][0], intri["cam_intrinsic"][1],
                intri["cam_intrinsic"][2], intri["cam_intrinsic"][3],
                intri["cam_intrinsic"][4], intri["cam_intrinsic"][5],
                intri["cam_intrinsic"][6], intri["cam_intrinsic"][7],
                intri["cam_intrinsic"][8]);
  cv::Vec<double, 8> dist(
      intri["cam_distcoeffs"][0], intri["cam_distcoeffs"][1],
      intri["cam_distcoeffs"][2], intri["cam_distcoeffs"][3],
      intri["cam_distcoeffs"][4], intri["cam_distcoeffs"][5],
      intri["cam_distcoeffs"][6], intri["cam_distcoeffs"][7]);
  auto calibrator = ChessboardCalibrator(K, dist);
  auto res = calibrator.detect(img_path, 1080, 1920, cv::Size(6, 3));
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
  test_single();
  return 0;
}
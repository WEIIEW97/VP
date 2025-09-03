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
#include "../src/json.h"
#include <boost/program_options.hpp>
#include <limits>

using namespace std;
namespace po = boost::program_options;

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

CameraParams load_json(const std::string& json_path) {
  auto intrinsic = read_json(json_path);
  cv::Matx33d K(intrinsic["cam_intrinsic"][0], intrinsic["cam_intrinsic"][1],
                intrinsic["cam_intrinsic"][2], intrinsic["cam_intrinsic"][3],
                intrinsic["cam_intrinsic"][4], intrinsic["cam_intrinsic"][5],
                intrinsic["cam_intrinsic"][6], intrinsic["cam_intrinsic"][7],
                intrinsic["cam_intrinsic"][8]);
  cv::Vec<double, 8> dist(
      intrinsic["cam_distcoeffs"][0], intrinsic["cam_distcoeffs"][1],
      intrinsic["cam_distcoeffs"][2], intrinsic["cam_distcoeffs"][3],
      intrinsic["cam_distcoeffs"][4], intrinsic["cam_distcoeffs"][5],
      intrinsic["cam_distcoeffs"][6], intrinsic["cam_distcoeffs"][7]);

  // if intrinsic has "model_type" field, then we can determine if it's fisheye
  bool is_fisheye = false;
  if (intrinsic.contains("model_type")) {
    std::string model_type = intrinsic["model_type"];
    if (model_type == "KANNALA_BRANDT") {
      is_fisheye = true;
    }
  }
  return {cv::Mat(K), dist, is_fisheye};
}

ChessboardCalibrator::CalibResult
psycho(const string& input_path, const string& intrinsic_path,
       int image_height = 1080, int image_width = 1920,
       const cv::Size& pattern_size = cv::Size(6, 3),
       float square_size = 0.08) {

  // auto cam_params = load_yaml(intrinsic_path);
  auto cam_params = load_json(intrinsic_path);

  auto calibrator = ChessboardCalibrator(cam_params.K, cam_params.dist_coef);
  auto res =
      calibrator.detect(input_path, image_height, image_width, pattern_size,
                        square_size, cam_params.is_fisheye);
  if (!res.success) {
    std::cerr << "Calibration failed!" << std::endl;
    return {false,
            cv::Vec3d(std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()),
            {}};
  }
  return res;
}

int main(int argc, char** argv) {
  string input_path, intrinsic_path;
  int image_height, image_width;
  cv::Size pattern_size;
  float square_size;

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "input,i", po::value<std::string>(&input_path)->required(),
      "Path to input .yuv or .png/.jpg.. file")(
      "intrinsic,k", po::value<std::string>(&intrinsic_path)->required(),
      "Path to intrinsic JSON file")(
      "height,h", po::value<int>(&image_height)->default_value(1080),
      "Image height")(
      "width,w", po::value<int>(&image_width)->default_value(1920),
      "Image width")("pattern_rows,r",
                     po::value<int>()->default_value(3)->notifier(
                         [&](int v) { pattern_size.height = v; }),
                     "Number of inner corners per chessboard row")(
      "pattern_cols,c",
      po::value<int>()->default_value(6)->notifier(
          [&](int v) { pattern_size.width = v; }),
      "Number of inner corners per chessboard column")(
      "square_size,s", po::value<float>(&square_size)->default_value(0.08f),
      "Size of a square in your defined unit (point, meter,etc.)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  auto calib_result = psycho(input_path, intrinsic_path, image_height,
                             image_width, pattern_size, square_size);
  if (calib_result.success) {
    cout << "Rotation angle (degrees): " << "\n"
         << "Around Z: " << calib_result.angle_degrees[0] << "\n"
         << "Around Y: " << calib_result.angle_degrees[1] << "\n"
         << "Around X: " << calib_result.angle_degrees[2] << endl;
  } else {
    cerr << "Failed to detect rotation angles, please check your "
            "conditions manually!"
         << endl;
    return -1;
  }
  return 0;
}
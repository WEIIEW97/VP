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
#include "src/json.h"
#include <boost/program_options.hpp>
#include <limits>

using json = nlohmann::json;
using namespace std;
namespace po = boost::program_options;

float psycho(const string& input_path, const string& intrinsic_path,
             int image_height = 1080, int image_width = 1920,
             const cv::Size& pattern_size = cv::Size(6, 3),
             float square_size = 0.025) {
  float rotation_angle_degrees = std::numeric_limits<float>::max();
  auto intrinsic = read_json(intrinsic_path);
  cv::Matx33d K(
      intrinsic["cam_intrinsicnsic"][0], intrinsic["cam_intrinsicnsic"][1],
      intrinsic["cam_intrinsicnsic"][2], intrinsic["cam_intrinsicnsic"][3],
      intrinsic["cam_intrinsicnsic"][4], intrinsic["cam_intrinsicnsic"][5],
      intrinsic["cam_intrinsicnsic"][6], intrinsic["cam_intrinsicnsic"][7],
      intrinsic["cam_intrinsicnsic"][8]);
  cv::Vec<double, 8> dist(
      intrinsic["cam_distcoeffs"][0], intrinsic["cam_distcoeffs"][1],
      intrinsic["cam_distcoeffs"][2], intrinsic["cam_distcoeffs"][3],
      intrinsic["cam_distcoeffs"][4], intrinsic["cam_distcoeffs"][5],
      intrinsic["cam_distcoeffs"][6], intrinsic["cam_distcoeffs"][7]);

  auto calibrator = ChessboardCalibrator(K, dist);
  auto res = calibrator.detect(input_path, image_height, image_width,
                               pattern_size, square_size);
  if (res.success) {
    rotation_angle_degrees = static_cast<float>(res.angle_degrees[1]);
  } else {
    std::cerr << "Failed to run calibration!" << std::endl;
  }
  return rotation_angle_degrees;
}

int main(int argc, char** argv) {
  std::string input_path, intrinsic_path;
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
      "square_size,s", po::value<float>(&square_size)->default_value(0.025f),
      "Size of a square in your defined unit (point, millimeter,etc.)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  float rotation_angle_degrees =
      psycho(input_path, intrinsic_path, image_height, image_width,
             pattern_size, square_size);
  if (rotation_angle_degrees != std::numeric_limits<float>::max()) {
    std::cout << "Rotation angle (degrees): " << rotation_angle_degrees
              << std::endl;
  } else {
    std::cerr << "Failed to detect rotation angles, please check your "
                 "conditions manually!"
              << std::endl;
    return -1;
  }
  return 0;
}
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

#include "e_recalib.h"

#include "recalib.h"

#include <stdexcept>
#include "../src/json.h"

using json = nlohmann::json;
using namespace std;

RecalibInfo recalib(const string& input_path, const string& intrinsic_path,
                    int image_height, int image_width,
                    const cv::Size& pattern_size, float square_size) {
  auto intrinsic = read_json(intrinsic_path);
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

  auto calibrator = ChessboardCalibrator(K, dist);
  auto res = calibrator.detect(input_path, image_height, image_width,
                               pattern_size, square_size);
  if (!res.success) {
    throw std::runtime_error("Recalibration failed!");
  }

  return {res.angle_degrees, K};
}

cv::Mat adjust(const RecalibInfo& info, const cv::Mat& im) {
  auto ypr = info.angle_degrees;
  auto h = im.rows;
  auto w = im.cols;

  auto R = ypr2R(ypr);
  auto H = info.K * R.inv() * info.K.inv();

  cv::Mat warped;
  cv::warpPerspective(im, warped, H, cv::Size(w, h));
  return warped;
}
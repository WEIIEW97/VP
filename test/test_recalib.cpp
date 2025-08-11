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

int main() {
  std::string yuv_path = "/home/william/Codes/vp/data/recalib/calib.yuv";
  cv::Matx33d K(1057.860222, 0, 977.840981, 0, 1059.252092, 572.529502, 0, 0,
                1);
  cv::Vec<double, 8> dist(0.47297, 0.482529, 0.000504, -0.000246, 0.042125,
                          0.877016, 0.602693, 0.192676);

  auto calibrator = ChessboardCalibrator(K, dist);
  auto res = calibrator.detect(yuv_path, 1080, 1920);
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
  return 0;
}
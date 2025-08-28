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

#include "../lib/e_recalib.h"

#include <iostream>
#include <string>

int main() {
  std::string img_path = "/home/william/extdisk/data/calib/image_save/abnor-63/"
                         "RGB/rgbvi-2025-8-22-12-2-16.png";
  std::string intri_path =
      "/home/william/extdisk/data/calib/image_save/calibration_intrinsic/"
      "abnor-63/result/intrinsics_colin.json";
  auto recalib_info = recalib(img_path, intri_path);
  std::cout << recalib_info.angle_degrees << std::endl;

  auto rgb = cv::imread(img_path);
  auto warped = adjust(recalib_info, rgb);
  cv::imshow("corrected image", warped);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}
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

#include <iostream>
#include "src/json.h"
#include <string>
#include "vp.h"

using namespace std;

int main() {
  string json_path = "/home/william/extdisk/data/Lane_Detection_Result/"
                     "20250219/19700101_002523_main.json";
  auto info = retrieve_info(json_path);
  int frame_id = 1;
  auto result = retrieve_pack_info_by_frame(info, frame_id);
  auto frame = json_to_eigen_matrix(result);
  Eigen::Matrix3f K;
  K << 1033.788708, 0, 916.010200, 0, 1033.780937, 522.486183, 0, 0, 1;
  Eigen::VectorXf dist_coefs(8);
  dist_coefs << 63.285886f, 34.709119f, 0.000035f, 0.000081f, 1.231907f,
      63.752675f, 61.351695f, 8.551888f;
  auto vp_detector = VP(K, dist_coefs);
  auto yp = vp_detector.get_yp_estimation(frame);
  cout << yp << endl;
  return 0;
}

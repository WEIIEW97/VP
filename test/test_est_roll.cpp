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

#include "vp.h"
#include <iostream>
#include <string>
#include "../src/json.h"

using namespace std;

int main() {
  const string info_path = "/home/william/Codes/vp/data/lanes/lane.json";
  const string img_dir = "/home/william/Codes/vp/data/lanes/samples";
  const string box_path = "/home/william/Codes/vp/data/lanes/person.json";
  const string intri_path =
      "/home/william/Codes/vp/data/lanes/intrinsics_colin.json";

  auto info = read_json(info_path);
  auto result = retrieve_pack_info_by_id(info, 180, "lane");
  auto frame = json_to_eigen_matrix(result);

  Eigen::Matrix3f K;
  Eigen::VectorXf dist_coefs(8);
  K << 1037.416932, 0, 974.447538, 0, 1038.440160, 564.996364, 0, 0, 1;
  dist_coefs << 2.263742, 5.859805, 0.001165, -0.000197, 0.522284, 2.592719,
      6.896840, 2.312912;
  
  auto vp_detector = VP(K, dist_coefs, 5, CameraModel::pinhole_k6, true);
  auto yp = vp_detector.get_yp_estimation(frame);
  cout << "Estimated yp: " << yp.transpose() << endl;

  Eigen::Vector2d uv1(981.60622304, 819.70624924);
  Eigen::Vector2d uv2(987.69220956, 732.07843781);
  double cam_h = 0.76;
  Eigen::Vector3d pw1(0, 3, 0);
  Eigen::Vector3d pw2(0, 5, 0);
  auto ypr = vp_detector.get_ypr_estimation(frame, uv1, uv2, pw1, pw2, cam_h);
  cout << "Estimated YPR: " << ypr.transpose() << endl;
  return 0;
}
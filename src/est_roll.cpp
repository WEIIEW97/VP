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

#include "est_roll.h"

#include <cmath>

PoseResult CameraPoseSolver::solve_from_two_points(const Eigen::Vector2d& uv1,
                                                   const Eigen::Vector2d& uv2,
                                                   const Eigen::Vector3d& pw1,
                                                   const Eigen::Vector3d& pw2,
                                                   double cam_h, double yaw,
                                                   double pitch) {
  PoseResult result;
  double roll =
      (last_roll_ == std::numeric_limits<double>::max()) ? 0.0f : last_roll_;
  ReprojectionErrorOptimizer optimizer(uv1, uv2, pw1, pw2, cam_h, yaw, pitch,
                                       K_);
  auto [best_roll, best_reproj_error] = optimizer.optimize();
  auto est_R = ypr2R(yaw, pitch, best_roll);
  auto T_b_c = Eigen::Vector3d(0, 0, -cam_h);
  result.roll = best_roll;
  result.R = est_R;
  result.T = T_b_c;
  result.reproj_error = best_reproj_error;
  return result;
}

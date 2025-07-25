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
#include <fmt/format.h>

Eigen::Matrix3f CameraPoseSolver::rotation_matrix(float yaw, float pitch, float roll) {
  Eigen::Matrix3f R;

  float cy = cos(yaw), sy = sin(yaw);
  float cp = cos(pitch), sp = sin(pitch);
  float cr = cos(roll), sr = sin(roll);

  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp,
      sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr;
  return R;
}

CameraPoseSolver::PoseResult CameraPoseSolver::solve_from_two_points(
    const Eigen::Vector2f& uv1, const Eigen::Vector2f& uv2,
    const Eigen::Vector3f& pw1, const Eigen::Vector3f& pw2, float cam_h,
    float yaw, float pitch) {
  PoseResult result;
  ceres::Problem problem;
  float roll = (last_roll_ == FLT_MAX) ? 0.0f : last_roll_;

  
  return result;
}
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

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include <cfloat>

class CameraPoseSolver {
public:
  struct PoseResult {
    float roll;
    Eigen::MatrixX3f R;
    Eigen::Vector3f T;
    float reproj_error;
  };

  CameraPoseSolver(const Eigen::MatrixX3f& K) {
    K_ = K;
    K_inv_ = K.inverse();
    fx_ = K(0, 0);
    fy_ = K(1, 1);
    cx_ = K(0, 2);
    cy_ = K(1, 2);
  }

  static Eigen::Matrix3f rotation_matrix(float yaw, float pitch, float roll);
  PoseResult solve_from_two_points(const Eigen::Vector2f& uv1,
                                   const Eigen::Vector2f& uv2,
                                   const Eigen::Vector3f& pw1,
                                   const Eigen::Vector3f& pw2, float cam_h,
                                   float yaw, float pitch);

private:
  Eigen::MatrixX3f K_;
  Eigen::MatrixX3f K_inv_;
  float fx_, fy_, cx_, cy_;

  float last_roll_ = FLT_MAX;
  Eigen::MatrixX3f last_R_;
  Eigen::Vector3f last_T_;
};

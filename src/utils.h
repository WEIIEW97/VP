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

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <Eigen/Dense>
#include <cmath>
#include <random>

template <typename T>
T rad2deg(T rad) {
  return rad / M_PI * 180.f;
}

template <typename T>
T deg2rag(T deg) {
  return deg / 180.f * M_PI;
}

inline Eigen::MatrixXi generate_sample_points(int n, int w, int h) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis_x(0, w - 1);
  std::uniform_int_distribution<> dis_y(0, h - 1);

  Eigen::MatrixXi samples(n, 2);
  for (int i = 0; i < n; i++) {
    samples(i, 0) = dis_x(gen);
    samples(i, 1) = dis_y(gen);
  }
  return samples;
}

template <typename T>
Eigen::Matrix<T, 3, 3> ypr2R(const T& yaw, const T& pitch, const T& roll) {
  // note that the input yaw, pitcn,roll should be radians
  // and should be in ground to camera coordinate system
  T cy = std::cos(yaw), sy = std::sin(yaw);
  T cp = std::cos(pitch), sp = std::sin(pitch);
  T cr = std::cos(roll), sr = std::sin(roll);

  Eigen::Matrix<T, 3, 3> R;
  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp,
      sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr;
  return R;
}

template <typename T>
Eigen::Matrix<T, 3, 3> ypr2R(const Eigen::Vector<T, 3>& ypr) {
  return ypr2R(ypr(0), ypr(1), ypr(2));
}

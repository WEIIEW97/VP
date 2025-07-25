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
#include <cmath>
#include <fmt/format.h>

bool VP::judge_valid(const std::vector<Eigen::MatrixXf>& frame_pts, int thr) {
  return frame_pts.size() >= thr;
}

Eigen::VectorXf VP::polyfit(const Eigen::VectorXf& x,
                            const Eigen::VectorXf& y) {
  int n = x.size();
  if (n < min_num_pts_) {
    if (verbose_) {
      fmt::print(
          "line fitting must have at least {} points, but get {} instead.\n",
          min_num_pts_, n);
    }
    return Eigen::Vector2f::Zero();
  }

  // normalize x for numerical stability
  auto mu = x.mean();
  auto sigma = std::sqrt((x.array() - mu).square().sum() / (x.size() - 1));
  const Eigen::VectorXf xn = (x.array() - mu) / (sigma + 1e-10);
  Eigen::MatrixXf A(n, 2);
  A.col(0) = xn;
  A.col(1) = Eigen::VectorXf::Ones(n);

  // solve linear system A^T * A * [m; c] = A^T * y
  // Eigen::VectorXf params = (A.transpose() * A).ldlt().solve(A.transpose() *
  // y);
  Eigen::Vector2f coeffs_linear =
      A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
  const Eigen::VectorXf pred_linear = A * coeffs_linear;
  auto r2_linear = 1 - (y - pred_linear).squaredNorm() /
                           (y.array() - y.mean()).square().sum();

  // solve quadratic system
  Eigen::MatrixXf A_quad(n, 3);
  A_quad.col(0) = xn.array().square();
  A_quad.col(1) = xn;
  A_quad.col(2) = Eigen::VectorXf::Ones(n);

  Eigen::Vector3f coeffs_quad =
      A_quad.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
  const Eigen::VectorXf pred_quad = A_quad * coeffs_quad;
  auto r2_quad =
      1 - (y - pred_quad).squaredNorm() / (y.array() - y.mean()).square().sum();

  if (r2_quad > r2_linear + r2_thr_) {
    if (verbose_) {
      fmt::print("Curved points detected! Skipping line fit.\n");
      return Eigen::Vector2f::Zero();
    }
  }

  // restore back to original scale
  coeffs_linear(0) = coeffs_linear(0) / sigma;
  coeffs_linear(1) = coeffs_linear(1) - coeffs_linear(0) * mu;

  return coeffs_linear;
}

void VP::line_fit(const std::vector<Eigen::MatrixXf>& frame_pts) {
  for (const auto& pts : frame_pts) {
    auto undist_pt = undistort_points<float>(cam_model_, pts, K_, dist_coef_);
    //  auto undist_pt = undistort_points<float>(pts, K_, dist_coef_);
    auto x = undist_pt.col(0);
    auto y = undist_pt.col(1);
    auto params = polyfit(x, y);
    if (params(0) != 0 && params(1) != 0) {
      param_lst_.push_back(params);
      Eigen::Vector3f homo(params(0), -1.f, params(1));
      homo_lst_.push_back(homo);
    }
    if (verbose_) {
      fmt::print("Fitted line: y={}x + {}\n", params(0), params(1));
    }
  }

  line_fit_flag_ = param_lst_.size() >= 2;
}

void VP::compute_vp() {
  auto sz = homo_lst_.size();
  for (int i = 0; i < sz - 1; i++) {
    for (int j = i + 1; j < sz; j++) {
      auto z = homo_lst_[i].cross(homo_lst_[j]);
      vps_.push_back(z / z(2));
    }
  }

  if (verbose_) {
    for (const auto& vp : vps_) {
      fmt::print("VP candidates are: x={}, y={}\n", vp(0), vp(1));
    }
  }
}

Eigen::Vector2f VP::filter_candidates(const std::string& strategy) {
  int full_len = vps_.size();
  Eigen::MatrixX3f vp_array3(full_len, 3);
  for (int i = 0; i < full_len; i++) {
    vp_array3.row(i) = vps_[i].transpose();
  }
  Eigen::MatrixX2f vp_array = vp_array3.leftCols(2);

  if (strategy == "mean") {
    return vp_array.colwise().mean().transpose();

  } else if (strategy == "close") {
    // if (vp_array.rows() < 2) {
    //   throw std::runtime_error(
    //       "At least 2 points required for 'close' strategy");
    // }

    Eigen::VectorXf vv = vp_array.col(0);
    int n = vv.size();

    Eigen::MatrixXf diff(n, n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        diff(i, j) = std::abs(vv[i] - vv[j]);
      }
      diff(i, i) = std::numeric_limits<float>::infinity();
    }

    Eigen::Index i, j;
    diff.minCoeff(&i, &j);

    Eigen::RowVector2f p1 = vp_array.row(i);
    Eigen::RowVector2f p2 = vp_array.row(j);
    Eigen::Vector2f result = (p1 + p2) / 2.0f;

    return result;

  } else {
    throw std::invalid_argument("Unknown strategy: " + strategy);
  }
}

Eigen::Vector2f VP::temporal_smooth(const Eigen::Vector2f& vp) {
  if (vp.isZero()) {
    if (vps_trace_.empty()) {
      return Eigen::Vector2f::Zero();
    }
    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    for (const auto& v : vps_trace_) {
      mean += v;
    }
    return mean / vps_trace_.size();
  }

  vps_trace_.push_back(vp);
  if (vps_trace_.size() > window_size_) {
    vps_trace_.pop_front();
  }
  Eigen::Vector2f mean = Eigen::Vector2f::Zero();
  for (const auto& v : vps_trace_) {
    mean += v;
  }
  return mean / vps_trace_.size();
}

Eigen::Vector2f VP::estimate_yp(const Eigen::Vector2f& vp) {
  auto vp_smooth = temporal_smooth(vp);
  float x, y;
  if (vp_smooth.isZero()) {
    x = vp(0);
    y = vp(1);
  } else {
    x = vp_smooth(0);
    y = vp_smooth(1);
  }
  auto yaw = std::atan((x - cx_) / fx_) * 180.f / M_PI;
  auto pitch = std::atan((cy_ - y) / fy_) * 180.f / M_PI;
  Eigen::Vector2f yp(yaw, pitch);
  return yp;
}

Eigen::Vector2f
VP::get_yp_estimation(const std::vector<Eigen::MatrixXf>& frame_pts) {
  if (!judge_valid(frame_pts)) {
    fmt::print("Not enough points to estimate vanishing point\n");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  line_fit(frame_pts);
  if (!line_fit_flag_) {
    fmt::print("Failed to fit lines\n");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  compute_vp();
  if (vps_.empty()) {
    fmt::print("No vanishing point candidates found\n");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  auto vp = filter_candidates("close");
  reload();
  return estimate_yp(vp);
}

Eigen::Vector2f
VP::get_vp_estimation(const std::vector<Eigen::MatrixXf>& frame_pts) {
  if (!judge_valid(frame_pts)) {
    fmt::print("Not enough points to estimate vanishing point");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  line_fit(frame_pts);
  if (!line_fit_flag_) {
    fmt::print("Failed to fit lines");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  compute_vp();
  if (vps_.empty()) {
    fmt::print("No vanishing point candidates found");
    return Eigen::Vector2f::Ones() * -99.f;
  }

  auto vp = filter_candidates("close");
  reload();
  return vp;
}

Eigen::Vector2f VP::get_yp_est_by_vp(const Eigen::Vector2f& vp) {
  return estimate_yp(vp);
}

void VP::reload() {
  param_lst_.clear();
  homo_lst_.clear();
  vps_.clear();
}

Eigen::Matrix3f VP::get_K() const { return K_; }

Eigen::VectorXf VP::get_dist() const { return dist_coef_; }
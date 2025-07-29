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
#include <fmt/format.h>

#include <limits>

#include "utils.h"

struct PoseResult {
  double roll;
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  double reproj_error;
};

class CameraPoseSolver {
public:
  CameraPoseSolver(const Eigen::Matrix3d& K) : K_(K) {}

  PoseResult solve_from_two_points(const Eigen::Vector2d& uv1,
                                   const Eigen::Vector2d& uv2,
                                   const Eigen::Vector3d& pw1,
                                   const Eigen::Vector3d& pw2, double cam_h,
                                   double yaw, double pitch);

private:
  Eigen::Matrix3d K_;
  double fx_, fy_, cx_, cy_;
  double last_roll_ = std::numeric_limits<double>::max();
  Eigen::Matrix3d
      last_R_; // [unused for now], saved for later stream data flow.
  Eigen::Vector3d last_T_;
};

class ReprojectionErrorOptimizer {
public:
  ReprojectionErrorOptimizer(const Eigen::Vector2d& uv1,
                             const Eigen::Vector2d& uv2,
                             const Eigen::Vector3d& Pw1,
                             const Eigen::Vector3d& Pw2, double h, double yaw,
                             double pitch, const Eigen::Matrix3d& K)
      : uv1_(uv1), uv2_(uv2), Pw1_(Pw1), Pw2_(Pw2), h_(h), K_(K), yaw_(yaw),
        pitch_(pitch) {
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.max_num_iterations = 100;
    options_.num_threads = 4;
    set_initial_guess();
  };

  struct CostFunctor {
    explicit CostFunctor(ReprojectionErrorOptimizer* optimizer)
        : optimizer_(optimizer) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
      T roll = params[0];
      Eigen::Matrix<T, 3, 3> R =
          ypr2R(optimizer_->yaw_, optimizer_->pitch_, roll);
      Eigen::Vector<T, 3> tvec(0, 0, -static_cast<T>(optimizer_->h_));

      Eigen::Vector<T, 3> Pc1 = R * (optimizer_->Pw1_.cast<T>() - tvec);
      Eigen::Vector<T, 3> Pc2 = R * (optimizer_->Pw2_.cast<T>() - tvec);

      Eigen::Vector<T, 3> uv1_reproj = optimizer_->K_.cast<T>() * Pc1;
      Eigen::Vector<T, 3> uv2_reproj = optimizer_->K_.cast<T>() * Pc2;
      uv1_reproj /= uv1_reproj(2);
      uv2_reproj /= uv2_reproj(2);

      // cache the last reprojection results
      optimizer_->last_uv1_reproj_ =
          uv1_reproj.template cast<double>().template head<2>();
      optimizer_->last_uv2_reproj_ =
          uv2_reproj.template cast<double>().template head<2>();

      residuals[0] = uv1_reproj(0) - static_cast<T>(optimizer_->uv1_(0));
      residuals[1] = uv1_reproj(1) - static_cast<T>(optimizer_->uv1_(1));
      residuals[2] = uv2_reproj(0) - static_cast<T>(optimizer_->uv2_(0));
      residuals[3] = uv2_reproj(1) - static_cast<T>(optimizer_->uv2_(1));
      return true;
    }

    ReprojectionErrorOptimizer* optimizer_;
  };

  void set_ceres_options(int max_num_iterations,
                         ceres::LinearSolverType linear_solver_type) {
    options_.max_num_iterations = max_num_iterations;
    options_.linear_solver_type = linear_solver_type;
  };

  void set_opt_num_of_threads(int n_threads) {
    options_.num_threads = n_threads;
  }

  void set_initial_guess(double lb = -M_PI / 6, double ub = M_PI / 6,
                         int n = 7) {
    initial_guess_.clear();
    initial_guess_.resize(n);
    double step = (ub - lb) / (n - 1);
    for (int i = 0; i < n; ++i) {
      initial_guess_[i] = lb + i * step;
    };
  }

  std::tuple<double, double> optimize_single(double init_roll) {
    ceres::Problem problem;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CostFunctor, 4, 1>(
                                 new CostFunctor(this)),
                             nullptr, &init_roll);

    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
      throw std::runtime_error("Optimization failed to converge");
    }

    fmt::print("{}\n", summary.FullReport());
    fmt::print("Estimated roll: {}\n", init_roll);

    auto reproj_uv1 = last_uv1_reproj_;
    auto reproj_uv2 = last_uv2_reproj_;
    auto reproj_error =
        (reproj_uv1 - uv1_).squaredNorm() + (reproj_uv2 - uv2_).squaredNorm();
    fmt::print("Reprojection error: {}\n", reproj_error);

    return {init_roll, reproj_error};
  };

  std::tuple<double, double> optimize() {
    double best_reproj_error = std::numeric_limits<double>::max();
    double best_roll = 0.0;
    for (const auto& init_roll : initial_guess_) {
      double roll = init_roll;
      auto [est_roll, reproj_error] = optimize_single(roll);
      if (reproj_error < best_reproj_error) {
        best_reproj_error = reproj_error;
        best_roll = est_roll;
      }
    }
    return {best_roll, best_reproj_error};
  }

private:
  Eigen::Vector2d uv1_, uv2_;
  Eigen::Vector3d Pw1_, Pw2_;
  double h_, yaw_, pitch_;
  Eigen::Matrix3d K_;
  bool is_deg_;
  ceres::Solver::Options options_;
  std::vector<double> initial_guess_;
  mutable Eigen::Vector2d last_uv1_reproj_;
  mutable Eigen::Vector2d last_uv2_reproj_;
};
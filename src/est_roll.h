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
#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>
#include <Eigen/Core>
#include <fmt/format.h>

#include <limits>

#include "utils.h"

struct PoseResult {
  double roll;
  Eigen::Matrix3d R_c_g;
  Eigen::Vector3d T_c_g;
  double reproj_error;
};

class CameraPoseSolver {
public:
  CameraPoseSolver(const Eigen::Matrix3d& K) : K_(K) {}

  PoseResult solve_from_two_points(const Eigen::Vector2d& uv1,
                                   const Eigen::Vector2d& uv2,
                                   const Eigen::Vector3d& pw1,
                                   const Eigen::Vector3d& pw2, double cam_h,
                                   double yaw_c_g, double pitch_c_g);

private:
  Eigen::Matrix3d K_;
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

  enum class CeresSolverMode {
    Auto,
    Tiny,
  };

  struct CostFunctor {
    explicit CostFunctor(ReprojectionErrorOptimizer* optimizer)
        : optimizer_(optimizer) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
      T roll = params[0];
      Eigen::Matrix<T, 3, 3> R =
          ypr2R(optimizer_->yaw_, optimizer_->pitch_,
                roll); // input yaw, pitch should be  ground to camera system
      Eigen::Vector<T, 3> tvec(0, 0, -static_cast<T>(optimizer_->h_));

      Eigen::Vector<T, 3> Pc1 = R * (optimizer_->Pw1_.cast<T>() - tvec);
      Eigen::Vector<T, 3> Pc2 = R * (optimizer_->Pw2_.cast<T>() - tvec);

      Eigen::Vector<T, 3> uv1_reproj =
          (optimizer_->K_.cast<T>() * Pc1) / (Pc1.z() + optimizer_->eps_);
      Eigen::Vector<T, 3> uv2_reproj =
          (optimizer_->K_.cast<T>() * Pc2) / (Pc1.z() + optimizer_->eps_);

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

    const double optimized_roll = init_roll;
    auto reproj_error = reproj_cost(optimized_roll);
    fmt::print("Reprojection error: {}\n", reproj_error);

    return {init_roll, reproj_error};
  };

  std::tuple<double, double> optimize_tiny(double init_roll) {
    CostFunctor cost_functor(this);
    using AutoDiffFunction =
        ceres::TinySolverAutoDiffFunction<CostFunctor, 4, 1>;
    AutoDiffFunction f(cost_functor);

    ceres::TinySolver<AutoDiffFunction> solver;
    solver.options.max_num_iterations = 50;
    solver.options.gradient_tolerance = 1e-6;
    solver.options.parameter_tolerance = 1e-6;

    Eigen::Matrix<double, 1, 1> roll;
    roll(0) = init_roll;
    solver.Solve(f, &roll);
    fmt::print("Estimated roll: {}\n", roll(0));

    const double optimized_roll = roll(0);
    auto reproj_error = reproj_cost(roll(0));
    fmt::print("Reprojection error: {}\n", reproj_error);

    return {roll(0), reproj_error};
  }

  std::tuple<double, double>
  optimize(CeresSolverMode mode = CeresSolverMode::Auto) {
    double best_reproj_error = std::numeric_limits<double>::max();
    double best_roll = 0.0;

    auto optimizer = [&](double init_roll) {
      auto [est_roll, reproj_error] = (mode == CeresSolverMode::Tiny)
                                          ? optimize_tiny(init_roll)
                                          : optimize_single(init_roll);

      if (reproj_error < best_reproj_error) {
        best_reproj_error = reproj_error;
        best_roll = est_roll;
      }
    };

    // Test all initial guesses
    for (double init_roll : initial_guess_) {
      optimizer(init_roll);
    }

    return {best_roll, best_reproj_error};
  }

private:
  Eigen::Vector2d uv1_, uv2_;
  Eigen::Vector3d Pw1_, Pw2_;
  double eps_ = 1e-8; // for numerical stability
  double h_, yaw_, pitch_;
  Eigen::Matrix3d K_;
  bool is_deg_;
  ceres::Solver::Options options_;
  std::vector<double> initial_guess_;
  Eigen::Vector2d last_uv1_reproj_;
  Eigen::Vector2d last_uv2_reproj_;

  double reproj_cost(double roll) {
    auto R = ypr2R(yaw_, pitch_, roll);
    auto tvec = Eigen::Vector3d(0, 0, -h_);
    Eigen::Vector3d Pc1 = R * (Pw1_.cast<double>() - tvec);
    Eigen::Vector3d Pc2 = R * (Pw2_.cast<double>() - tvec);
    last_uv1_reproj_ = (K_ * Pc1) / (Pc1.z() + eps_);
    last_uv2_reproj_ = (K_ * Pc2) / (Pc2.z() + eps_);
    auto reproj_uv1 = last_uv1_reproj_;
    auto reproj_uv2 = last_uv2_reproj_;
    return (reproj_uv1 - uv1_).squaredNorm() +
           (reproj_uv2 - uv2_).squaredNorm();
  }
};
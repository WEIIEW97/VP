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

#include <Eigen/Dense>
#include <cmath>

template <typename EigenScalarType>
Eigen::Matrix<EigenScalarType, -1, 2>
undistort_points(const Eigen::Matrix<EigenScalarType, -1, 2>& distorted_points,
                 const Eigen::Matrix<EigenScalarType, 3, 3>& K,
                 const Eigen::Vector<EigenScalarType, 8>& dist_coefs,
                 int max_iter = 8, EigenScalarType epsilon = 1e-7) {
  using Eigen::Array;
  using Eigen::Dynamic;
  using Eigen::Matrix;

  const int num_points = distorted_points.rows();
  Matrix<EigenScalarType, -1, 2> undistorted_points(num_points, 2);

  // Extract intrinsic parameters
  const EigenScalarType fx = K(0, 0);
  const EigenScalarType fy = K(1, 1);
  const EigenScalarType cx = K(0, 2);
  const EigenScalarType cy = K(1, 2);

  // Extract distortion coefficients
  const EigenScalarType k1 = dist_coefs[0], k2 = dist_coefs[1],
                        k3 = dist_coefs[4];
  const EigenScalarType k4 = dist_coefs[5], k5 = dist_coefs[6],
                        k6 = dist_coefs[7];
  const EigenScalarType p1 = dist_coefs[2], p2 = dist_coefs[3];

  const Array<EigenScalarType, Dynamic, 1> xn =
      (distorted_points.col(0).array() - cx) / fx;
  const Array<EigenScalarType, Dynamic, 1> yn =
      (distorted_points.col(1).array() - cy) / fy;

  Array<EigenScalarType, Dynamic, 1> xu = xn;
  Array<EigenScalarType, Dynamic, 1> yu = yn;

  // Newton-Raphson iteration
  for (int iter = 0; iter < max_iter; ++iter) {
    const Array<EigenScalarType, Dynamic, 1> x = xu;
    const Array<EigenScalarType, Dynamic, 1> y = yu;
    const Array<EigenScalarType, Dynamic, 1> r2 = x.square() + y.square();
    const Array<EigenScalarType, Dynamic, 1> r4 = r2.square();
    const Array<EigenScalarType, Dynamic, 1> r6 = r4 * r2;

    // Radial distortion components
    const Array<EigenScalarType, Dynamic, 1> num_radial =
        1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    const Array<EigenScalarType, Dynamic, 1> den_radial =
        1.0 + r2 * (k4 + r2 * (k5 + r2 * k6));
    const Array<EigenScalarType, Dynamic, 1> radial = num_radial / den_radial;

    // Tangential distortion components
    const Array<EigenScalarType, Dynamic, 1> xy = x * y;
    const Array<EigenScalarType, Dynamic, 1> dx_tang =
        2 * p1 * xy + p2 * (r2 + 2 * x.square());
    const Array<EigenScalarType, Dynamic, 1> dy_tang =
        2 * p2 * xy + p1 * (r2 + 2 * y.square());

    const Array<EigenScalarType, Dynamic, 1> x_distorted = x * radial + dx_tang;
    const Array<EigenScalarType, Dynamic, 1> y_distorted = y * radial + dy_tang;
    const Array<EigenScalarType, Dynamic, 1> x_err = xn - x_distorted;
    const Array<EigenScalarType, Dynamic, 1> y_err = yn - y_distorted;

    // Radial derivatives
    const Array<EigenScalarType, Dynamic, 1> dr_num =
        2 * k1 * x + 4 * k2 * x * r2 + 6 * k3 * x * r4;
    const Array<EigenScalarType, Dynamic, 1> dr_den =
        2 * k4 * x + 4 * k5 * x * r2 + 6 * k6 * x * r4;
    const Array<EigenScalarType, Dynamic, 1> drdx =
        (dr_num * den_radial - num_radial * dr_den) / den_radial.square();

    const Array<EigenScalarType, Dynamic, 1> dr_num_y =
        2 * k1 * y + 4 * k2 * y * r2 + 6 * k3 * y * r4;
    const Array<EigenScalarType, Dynamic, 1> dr_den_y =
        2 * k4 * y + 4 * k5 * y * r2 + 6 * k6 * y * r4;
    const Array<EigenScalarType, Dynamic, 1> drdy =
        (dr_num_y * den_radial - num_radial * dr_den_y) / den_radial.square();

    // Tangential derivatives
    const Array<EigenScalarType, Dynamic, 1> dtdx_x = 2 * p1 * y + 4 * p2 * x;
    const Array<EigenScalarType, Dynamic, 1> dtdx_y = 2 * p1 * x + 2 * p2 * y;
    const Array<EigenScalarType, Dynamic, 1> dtdy_x = 2 * p2 * y + 2 * p1 * x;
    const Array<EigenScalarType, Dynamic, 1> dtdy_y = 4 * p1 * y + 2 * p2 * x;

    // Jacobian matrix components
    const Array<EigenScalarType, Dynamic, 1> J11 = radial + x * drdx + dtdx_x;
    const Array<EigenScalarType, Dynamic, 1> J12 = x * drdy + dtdx_y;
    const Array<EigenScalarType, Dynamic, 1> J21 = y * drdx + dtdy_x;
    const Array<EigenScalarType, Dynamic, 1> J22 = radial + y * drdy + dtdy_y;

    const Array<EigenScalarType, Dynamic, 1> det = J11 * J22 - J12 * J21;
    const Array<EigenScalarType, Dynamic, 1> inv_det = 1.0 / det.max(epsilon);

    xu += inv_det * (J22 * x_err - J12 * y_err);
    yu += inv_det * (-J21 * x_err + J11 * y_err);

    if ((x_err.abs().maxCoeff() < epsilon) &&
        (y_err.abs().maxCoeff() < epsilon)) {
      break;
    }
  }

  undistorted_points.col(0) = xu * fx + cx;
  undistorted_points.col(1) = yu * fy + cy;

  return undistorted_points;
}

class VP {
public:
  VP(const Eigen::Matrix3f& K, const Eigen::VectorXf& dist_coef,
     int min_num_pts = 10, bool verbose = false)
      : K_(K), dist_coef_(dist_coef), min_num_pts_(min_num_pts),
        verbose_(verbose) {
    fx_ = K(0, 0);
    cx_ = K(0, 2);
    fy_ = K(1, 1);
    cy_ = K(1, 2);
  }

private:
  bool judge_valid(const std::vector<Eigen::MatrixXf>& frame_pts, int thr = 2);
  Eigen::VectorXf polyfit(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
  void line_fit(const std::vector<Eigen::MatrixXf>& frame_pts);
  void compute_vp();
  Eigen::Vector2f filter_candidates(const std::string& strategy);
  Eigen::Vector2f estimate_yp(const Eigen::Vector2f& vp);
  void reload();

public:
  Eigen::Vector2f
  get_yp_estimation(const std::vector<Eigen::MatrixXf>& frame_pts);

private:
  Eigen::Matrix3f K_;
  Eigen::VectorXf dist_coef_;
  int min_num_pts_;
  bool verbose_;
  std::vector<Eigen::Vector2f> param_lst_;
  std::vector<Eigen::Vector3f> homo_lst_;
  std::vector<Eigen::Vector3f> vps_;
  float fx_, cx_, fy_, cy_;
  bool line_fit_flag_ = true;
};
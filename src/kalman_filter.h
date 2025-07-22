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

// kalman-filter header-only implementation

#include <Eigen/Dense>

template <int StateDim, int MeasDim>
class KalmanFilter {
public:
  using StateVec =
      Eigen::Matrix<float, StateDim, 1>; // State vector (e.g., [x, y, vx, vy])
  using StateCov = Eigen::Matrix<float, StateDim, StateDim>; // State covariance
  using MeasVec =
      Eigen::Matrix<float, MeasDim, 1>; // Measurement vector (e.g., [x, y])
  using MeasCov =
      Eigen::Matrix<float, MeasDim, MeasDim>; // Measurement covariance
  using StateTransMat =
      Eigen::Matrix<float, StateDim, StateDim>; // State transition matrix (F)
  using MeasMat =
      Eigen::Matrix<float, MeasDim, StateDim>; // Measurement matrix (H)
  using KalmanGain = Eigen::Matrix<float, StateDim, MeasDim>; // Kalman gain (K)

  KalmanFilter(const StateTransMat& F, const MeasMat& H, const StateCov& Q,
               const MeasCov& R,
               const StateVec& initial_state = StateVec::Zero(),
               const StateCov& initial_covariance = StateCov::Identity())
      : F_(F), H_(H), Q_(Q), R_(R), x_(initial_state), P_(initial_covariance),
        I_state_(StateCov::Identity()) {}

  /**
   * @brief Predicts the next state and covariance.
   * x_k|k-1 = F * x_k-1|k-1
   * P_k|k-1 = F * P_k-1|k-1 * F^T + Q
   */
  void predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
  }

  void update(const MeasVec& z) {
    const MeasVec y = z - H_ * x_; //  y_k = z_k - H * x_k|k-1
    const MeasCov S =
        H_ * P_ * H_.transpose() +
        R_; // Innovation (or residual) covariance: S_k = H * P_k|k-1 * H^T + R
    // const KalmanGain K = P_ * H_.transpose() * S.inverse();
    const KalmanGain K =
        (S.ldlt().solve((P_ * H_.transpose()).transpose()))
            .transpose(); // LDLT decomposition for numerical stability for K =
                          // P_k|k-1 * H^T * S^{-1}
    x_ = x_ + K * y;

    // P_ = (StateCov::Identity() - K * H_) * P_;

    // Updated error covariance using Joseph form for numerical stability:
    // P_k|k = (I - K_k * H) * P_k|k-1 * (I - K_k * H)^T + K_k * R * K_k^T
    // Or the simpler (I - K_k * H) * P_k|k-1 if P_ is P_k|k-1
    // Let's use the Joseph form for better numerical stability
    StateCov I_KH = I_state_ - K * H_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
  }

  StateVec predict_and_update(const MeasVec& z) {
    predict();
    update(z);
    return x_;
  }

  const StateVec& state() const { return x_; }
  const StateCov& covariance() const { return P_; }
  void set_f(const StateTransMat& F) { F_ = F; }
  void set_h(const MeasMat& H) { H_ = H; }
  void set_q(const StateCov& Q) { Q_ = Q; }
  void set_r(const MeasCov& R) { R_ = R; }

private:
  StateTransMat F_;
  MeasMat H_;
  StateCov Q_;
  MeasCov R_;
  StateVec x_;
  StateCov P_;

  StateCov I_state_;
};


class KalmanFilterTracker {
public:
  KalmanFilterTracker(float p_noise = 0.1f, float m_noise = 0.5f)
      : kf_(F_, H_, Q_, R_) {
    // State transition [x, y, vx, vy]
    F_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
    // Measurement [x, y]
    H_ << 1, 0, 0, 0, 0, 1, 0, 0;
    Q_ = StateCov::Identity() * p_noise;
    R_ = MeasCov::Identity() * m_noise;
  }

  Eigen::Vector2f update(const Eigen::Vector2f& vp) {

    kf_.predict();
    kf_.update(vp);
    return kf_.state().template head<2>();
  }

private:
  using StateCov = Eigen::Matrix<float, 4, 4>;
  using MeasCov = Eigen::Matrix<float, 2, 2>;
  KalmanFilter<4, 2> kf_;
  Eigen::Matrix4f F_;            // State transition
  Eigen::Matrix<float, 2, 4> H_; // Measurement
  StateCov Q_;                   // Process noise
  MeasCov R_;                    // Measurement noise
};

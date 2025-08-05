// Inverse Perspective Mapping (Bird-eye View)
// Copyright (c) 2025, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx
#ifndef IPM_H_
#define IPM_H_

#include <opencv2/core.hpp>
#include <numbers>

class IPM {
public:
  struct IPMInfo {
    /// conversion between mm in world coordinate on the ground in x-direction
    /// and pixel in image
    float x_scale;
    /// conversion between mm in world coordinate on the ground in y-direction
    /// and pixel in image
    float y_scale;

    // set the IPM image width
    int width;
    // set the IPM image height
    int height;

    // portion of image height to add to y-coordinate of vanishing point
    float vp_portion;

    // Left boundary in original image of region to make IPM for //ROI?
    int boundary_left;
    // Right boundary in original image of region to make IPM for //ROI?
    int boundary_right;
    // Top boundary in original image of region to make IPM for //ROI?
    int boundary_top;
    // Bottom boundary in original image of region to make IPM for //ROI?
    int boundary_bottom;

    IPMInfo() {
      x_scale = 100;
      y_scale = 100;
      width = 1000;
      height = 1000;
      vp_portion = 0.15;
      boundary_left = 20;
      boundary_right = 20;
      boundary_top = 20;
      boundary_bottom = 20;
    }
  };

  IPM() = delete;
  explicit IPM(cv::Matx33d& K_src, cv::Vec<double, 8>& dist, cv::Size img_size,
               double yaw_c_b, double pitch_c_b, double roll_c_b, double tx_b_c,
               double ty_b_c, double tz_b_c, bool is_fisheye = false);

  IPM(const cv::Matx33d& K_src, const cv::Vec<double, 8>& dist,
      const cv::Size& img_size, const cv::Matx33d& R_c_b,
      const cv::Vec3d& t_c_b, bool is_fisheye = false);

  cv::Point2d ProjectPointUV2BEVXY(const cv::Point2d& uv, cv::Matx33d& R_c_g);
  cv::Point2d ProjectPointUV2BEVXY(const cv::Point2d& uv, double yaw_c_g,
                                   double pitch_c_g, double roll_c_g);
  std::vector<cv::Point2d>
  ProjectPointUV2BEVXY(const std::vector<cv::Point2d>& uv, cv::Matx33d& R_c_g);
  std::vector<cv::Point2d>
  ProjectPointUV2BEVXY(const std::vector<cv::Point2d>& uv, double yaw_c_g,
                       double pitch_c_g, double roll_c_g);

  cv::Point2d ProjectBEVXY2PointUV(const cv::Point2d& xy, cv::Matx33d& R_c_g);
  cv::Point2d ProjectBEVXY2PointUV(const cv::Point2d& xy, double yaw_c_g,
                                   double pitch_c_g, double roll_c_g);
  std::vector<cv::Point2d>
  ProjectBEVXY2PointUV(const std::vector<cv::Point2d>& xy, cv::Matx33d& R_c_g);
  std::vector<cv::Point2d>
  ProjectBEVXY2PointUV(const std::vector<cv::Point2d>& xy, double yaw_c_g,
                       double pitch_c_g, double roll_c_g);

  double EstimateImuPitchOffset(const cv::Point2d& marker_uv,
                                const cv::Point2d& marker_xy, double yaw_c_g,
                                double pitch_c_g, double roll_c_g,
                                double h_g_c);

  cv::Mat GetIPMImage(cv::Mat& image, IPMInfo ipm_info, cv::Matx33d& R_c_g);
  cv::Mat GetIPMImage(cv::Mat& image, IPMInfo ipm_info, double yaw_c_g,
                      double pitch_c_g, double roll_c_g);

private:
  std::vector<cv::Vec3d> LimitBEVLine(cv::Matx33d& R_c_g, cv::Vec3d& t_c_g);
  double LimitBEVy(const std::vector<cv::Vec3d>& line, const cv::Point2d& xy);

  static cv::Matx33d TransformImage2Ground(cv::Matx33d& R_c_g,
                                           cv::Vec3d& t_c_g);
  static cv::Matx33d TransformGround2Image(cv::Matx33d& R_c_g,
                                           cv::Vec3d& t_c_g);

  static cv::Vec3d R2YPR(const cv::Matx33d& R);
  static cv::Matx33d YPR2R(const cv::Vec3d& ypr);

  template <typename T>
  static T r(T k2, T k3, T k4, T k5, T theta) {
    // k1 = 1
    return theta + k2 * theta * theta * theta +
           k3 * theta * theta * theta * theta * theta +
           k4 * theta * theta * theta * theta * theta * theta * theta +
           k5 * theta * theta * theta * theta * theta * theta * theta * theta *
               theta;
  }

  static cv::Point2d SpaceToPlane(const cv::Vec3d& p3d, const cv::Matx33d& K,
                                  cv::Vec<double, 8>& D, bool is_fisheye);

  cv::Matx33d K_src_;
  cv::Vec<double, 8> dist_;
  cv::Size img_size_;
  cv::Matx33d K_dst_, K_dst_inv_, R_c_b_;
  cv::Vec3d t_c_b_;
  bool is_fisheye_;
};

#endif // IPM_H_
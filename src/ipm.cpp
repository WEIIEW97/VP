// Inverse Perspective Mapping (Bird-eye View)
// Copyright (c) 2025, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include "ipm.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

IPM::IPM(cv::Matx33d& K_src, cv::Vec<double, 8>& dist, cv::Size img_size,
         double yaw_c_b, double pitch_c_b, double roll_c_b, double tx_b_c,
         double ty_b_c, double tz_b_c, bool is_fisheye)
    : K_src_(K_src), dist_(dist), img_size_(img_size), is_fisheye_(is_fisheye) {
  if (!is_fisheye_)
    K_dst_ = cv::getOptimalNewCameraMatrix(K_src_, dist_, img_size_, -1,
                                           img_size_, (cv::Rect*)nullptr, true);
  else
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K_src, cv::Mat(dist_).rowRange(0, 4), img_size_, cv::Matx33d::eye(),
        K_dst_);

  K_dst_inv_ = K_dst_.inv();
  cv::Matx33d R_c_b = YPR2R({yaw_c_b, pitch_c_b, roll_c_b});
  cv::Vec3d t_b_c(tx_b_c, ty_b_c, tz_b_c);
  t_c_b_ = -R_c_b * t_b_c;
}

IPM::IPM(const cv::Matx33d& K_src, const cv::Vec<double, 8>& dist,
         const cv::Size& img_size, const cv::Matx33d& R_c_b,
         const cv::Vec3d& t_c_b, bool is_fisheye)
    : K_src_(K_src), dist_(dist), img_size_(img_size), is_fisheye_(is_fisheye) {
  if (!is_fisheye_)
    K_dst_ = cv::getOptimalNewCameraMatrix(K_src_, dist_, img_size_, -1,
                                           img_size_, (cv::Rect*)nullptr, true);
  else
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K_src, cv::Mat(dist_).rowRange(0, 4), img_size_, cv::Matx33d::eye(),
        K_dst_);
  R_c_b_ = R_c_b;
  t_c_b_ = t_c_b;
}

cv::Point2d IPM::ProjectPointUV2BEVXY(const cv::Point2d& uv,
                                      cv::Matx33d& R_c_g) {
  std::vector<cv::Point2d> uv_src{uv};
  std::vector<cv::Point2d> uv_dst;
  if (!is_fisheye_)
    cv::undistortPoints(uv_src, uv_dst, K_src_, dist_, K_dst_);
  else
    cv::fisheye::undistortPoints(uv_src, uv_dst, K_dst_,
                                 cv::Mat(dist_).rowRange(0, 4), cv::Mat(),
                                 K_dst_);

  cv::Matx33d H_g_c = TransformImage2Ground(R_c_g, t_c_b_);
  H_g_c = H_g_c * K_dst_inv_;
  //  H_g_c /= H_g_c(2, 2);

  cv::Vec3d point_uv(uv_dst.at(0).x, uv_dst.at(0).y, 1);
  cv::Vec3d point_3d = H_g_c * point_uv;
  point_3d /= point_3d(2);
  return cv::Point2d(point_3d(0), point_3d(1));
}

cv::Point2d IPM::ProjectPointUV2BEVXY(const cv::Point2d& uv, double yaw_c_g,
                                      double pitch_c_g, double roll_c_g) {
  cv::Matx33d R_c_g = YPR2R({yaw_c_g, pitch_c_g, roll_c_g});
  return ProjectPointUV2BEVXY(uv, R_c_g);
}

std::vector<cv::Point2d>
IPM::ProjectPointUV2BEVXY(const std::vector<cv::Point2d>& uv,
                          cv::Matx33d& R_c_g) {
  std::vector<cv::Point2d> uv_dst;
  if (!is_fisheye_)
    cv::undistortPoints(uv, uv_dst, K_src_, dist_, K_dst_);
  else
    cv::fisheye::undistortPoints(
        uv, uv_dst, K_dst_, cv::Mat(dist_).rowRange(0, 4), cv::Mat(), K_dst_);

  cv::Matx33d H_g_c = TransformImage2Ground(R_c_g, t_c_b_);
  H_g_c = H_g_c * K_dst_inv_;
  //  H_g_c /= H_g_c(2, 2);

  std::vector<cv::Point2d> rst;
  rst.reserve(uv_dst.size());
  for (auto& item : uv_dst) {
    cv::Vec3d point_uv(item.x, item.y, 1);
    cv::Vec3d point_3d = H_g_c * point_uv;
    point_3d /= point_3d(2);
    rst.emplace_back(cv::Point2d(point_3d(0), point_3d(1)));
  }

  return rst;
}

std::vector<cv::Point2d>
IPM::ProjectPointUV2BEVXY(const std::vector<cv::Point2d>& uv, double yaw_c_g,
                          double pitch_c_g, double roll_c_g) {
  cv::Matx33d R_c_g = YPR2R({yaw_c_g, pitch_c_g, roll_c_g});
  return ProjectPointUV2BEVXY(uv, R_c_g);
}

cv::Point2d IPM::ProjectBEVXY2PointUV(const cv::Point2d& xy,
                                      cv::Matx33d& R_c_g) {
  auto line = LimitBEVLine(R_c_g, t_c_b_);

  cv::Vec3d bev_xy(xy.x, LimitBEVy(line, xy), 1);
  cv::Matx33d H_i_g = TransformGround2Image(R_c_g, t_c_b_);
  cv::Vec3d xyz_i = H_i_g * bev_xy;
  cv::Point2d uv = SpaceToPlane(xyz_i, K_src_, dist_, is_fisheye_);

  return uv;
}

cv::Point2d IPM::ProjectBEVXY2PointUV(const cv::Point2d& xy, double yaw_c_g,
                                      double pitch_c_g, double roll_c_g) {
  cv::Matx33d R_c_g = YPR2R({yaw_c_g, pitch_c_g, roll_c_g});
  return ProjectBEVXY2PointUV(xy, R_c_g);
}

std::vector<cv::Point2d>
IPM::ProjectBEVXY2PointUV(const std::vector<cv::Point2d>& xy,
                          cv::Matx33d& R_c_g) {
  auto line = LimitBEVLine(R_c_g, t_c_b_);
  cv::Matx33d H_i_g = TransformGround2Image(R_c_g, t_c_b_);

  std::vector<cv::Point2d> rst;
  rst.reserve(xy.size());

  for (auto& item : xy) {
    cv::Vec3d bev_xy(item.x, LimitBEVy(line, item), 1);
    cv::Vec3d xyz_i = H_i_g * bev_xy;
    cv::Point2d uv = SpaceToPlane(xyz_i, K_src_, dist_, is_fisheye_);
    rst.emplace_back(uv);
  }

  return rst;
}

std::vector<cv::Point2d>
IPM::ProjectBEVXY2PointUV(const std::vector<cv::Point2d>& xy, double yaw_c_g,
                          double pitch_c_g, double roll_c_g) {
  cv::Matx33d R_c_g = YPR2R({yaw_c_g, pitch_c_g, roll_c_g});
  return ProjectBEVXY2PointUV(xy, R_c_g);
}

double IPM::EstimateImuPitchOffset(const cv::Point2d& marker_uv,
                                   const cv::Point2d& marker_xy, double yaw_c_g,
                                   double pitch_c_g, double roll_c_g,
                                   double h_g_c) {
  std::vector<cv::Point2d> uv_src{marker_uv};
  std::vector<cv::Point2d> uv_dst;
  if (!is_fisheye_)
    cv::undistortPoints(uv_src, uv_dst, K_src_, dist_, K_dst_);
  else
    cv::fisheye::undistortPoints(uv_src, uv_dst, K_dst_,
                                 cv::Mat(dist_).rowRange(0, 4), cv::Mat(),
                                 K_dst_);
  double beta = atan2(uv_dst.at(0).y - K_dst_(1, 2), K_dst_(1, 1)) * 180 / std::numbers::pi;
  double alpha = atan2(marker_xy.y, h_g_c) * 180 / std::numbers::pi;
  double pitch_measurement = 180 - alpha - beta;
  double pitch_imu_est = roll_c_g;
  auto pitch_offset = pitch_measurement - pitch_imu_est;
  return pitch_offset;
}

cv::Mat IPM::GetIPMImage(cv::Mat& image, IPM::IPMInfo ipm_info,
                         cv::Matx33d& R_c_g) {
  cv::Matx33d K_g(ipm_info.x_scale, 0, 0.5 * (ipm_info.width - 1), 0,
                  -ipm_info.y_scale, ipm_info.height - 1, 0, 0, 1);
  cv::Size ipm_size(ipm_info.width, ipm_info.height);

  cv::Matx33d H_i_g = TransformGround2Image(R_c_g, t_c_b_);
  H_i_g = K_dst_ * H_i_g * K_g.inv();

  cv::Mat img_undist;
  if (!is_fisheye_)
    cv::undistort(image, img_undist, K_src_, dist_, K_dst_);
  else
    cv::fisheye::undistortImage(image, img_undist, K_src_,
                                cv::Mat(dist_).rowRange(0, 4), K_dst_);
  cv::Mat ipm_img;
  if (1) {
    cv::Mat map_x(ipm_size, CV_32F);
    cv::Mat map_y(ipm_size, CV_32F);
    for (int i = 0; i < ipm_size.height; ++i) {
      for (int j = 0; j < ipm_size.width; ++j) {
        cv::Vec3d xy(j, i, 1);
        cv::Vec3d uv = H_i_g * xy;
        uv /= uv(2);
        float col = uv(0), row = uv(1);
        map_x.at<float>(i, j) = col;
        map_y.at<float>(i, j) = row;
      }
    }

    cv::remap(img_undist, ipm_img, map_x, map_y, cv::INTER_LINEAR);
  } else {
    cv::warpPerspective(img_undist, ipm_img, H_i_g.inv(), ipm_size);
  }

  return ipm_img;
}

cv::Mat IPM::GetIPMImage(cv::Mat& image, IPM::IPMInfo ipm_info, double yaw_c_g,
                         double pitch_c_g, double roll_c_g) {
  cv::Matx33d R_c_g = YPR2R({yaw_c_g, pitch_c_g, roll_c_g});
  return GetIPMImage(image, ipm_info, R_c_g);
}

std::vector<cv::Vec3d> IPM::LimitBEVLine(cv::Matx33d& R_c_g, cv::Vec3d& t_c_g) {
  std::vector<cv::Point2d> bot_limit{
      cv::Point2d(0, img_size_.height * 1.5),
      cv::Point2d(img_size_.width * 0.5, img_size_.height * 1.5),
      cv::Point2d(img_size_.width, img_size_.height * 1.5),
  };
  bot_limit = ProjectPointUV2BEVXY(bot_limit, R_c_g);

  cv::Vec3d a(bot_limit.at(0).x, bot_limit.at(0).y, 1);
  cv::Vec3d b(bot_limit.at(1).x, bot_limit.at(1).y, 1);
  cv::Vec3d c(bot_limit.at(2).x, bot_limit.at(2).y, 1);

  cv::Vec3d line1 = a.cross(b);
  cv::Vec3d line2 = b.cross(c);

  return {line1, line2, cv::Vec3d(a(1), b(1), c(1))};
}

double IPM::LimitBEVy(const std::vector<cv::Vec3d>& line,
                      const cv::Point2d& xy) {
  double y = xy.y;
  double limit_y1 = -(line.at(0)(0) * xy.x + line.at(0)(2)) / line.at(0)(1);
  double limit_y2 = -(line.at(1)(0) * xy.x + line.at(1)(2)) / line.at(1)(1);

  auto limit_y = std::max(std::max(limit_y1, line.at(2)(0)),
                          std::max(limit_y2, line.at(2)(2)));
  return y > limit_y ? y : limit_y;
}

cv::Matx33d IPM::TransformImage2Ground(cv::Matx33d& R_c_g, cv::Vec3d& t_c_g) {
  cv::Matx33d H = TransformGround2Image(R_c_g, t_c_g).inv();
  return H;
}

cv::Matx33d IPM::TransformGround2Image(cv::Matx33d& R_c_g, cv::Vec3d& t_c_g) {
  cv::Matx33d H(R_c_g(0, 0), R_c_g(0, 1), t_c_g(0), R_c_g(1, 0), R_c_g(1, 1),
                t_c_g(1), R_c_g(2, 0), R_c_g(2, 1), t_c_g(2));
  return H;
}

cv::Vec3d IPM::R2YPR(const cv::Matx33d& R) {
  cv::Vec3d n(R(0, 0), R(1, 0), R(2, 0));
  cv::Vec3d o(R(0, 1), R(1, 1), R(2, 1));
  cv::Vec3d a(R(0, 2), R(1, 2), R(2, 2));

  cv::Vec3d ypr;
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r =
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));

  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / std::numbers::pi * 180.0;
}

cv::Matx33d IPM::YPR2R(const cv::Vec3d& ypr) {
  auto y = ypr(0) / 180.0 * std::numbers::pi;
  auto p = ypr(1) / 180.0 * std::numbers::pi;
  auto r = ypr(2) / 180.0 * std::numbers::pi;

  cv::Matx33d Rz(cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1);

  cv::Matx33d Ry(cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p));

  cv::Matx33d Rx(1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r));

  return Rz * Ry * Rx;
}

cv::Point2d IPM::SpaceToPlane(const cv::Vec3d& p3d, const cv::Matx33d& K,
                              cv::Vec<double, 8>& D, bool is_fisheye) {
  if (!is_fisheye) {
    cv::Point2d p_u, p_d;

    // Project points to the normalised plane
    p_u = cv::Point2d(p3d(0) / p3d(2), p3d(1) / p3d(2));

    // project 3D object point to the image plane
    double k1 = D(0);
    double k2 = D(1);
    double k3 = D(4);
    double k4 = D(5);
    double k5 = D(6);
    double k6 = D(7);
    double p1 = D(2);
    double p2 = D(3);

    // Transform to model plane
    double x = p_u.x;
    double y = p_u.y;

    double r2 = x * x + y * y;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double a1 = 2 * x * y;
    double a2 = r2 + 2 * x * x;
    double a3 = r2 + 2 * y * y;
    double cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
    double icdist2 = 1. / (1 + k4 * r2 + k5 * r4 + k6 * r6);

    cv::Point2d d_u(x * cdist * icdist2 + p1 * a1 + p2 * a2 - x,
                    y * cdist * icdist2 + p1 * a3 + p2 * a1 - y);
    p_d = p_u + d_u;

    cv::Point2d uv(K(0, 0) * p_d.x + K(0, 2), K(1, 1) * p_d.y + K(1, 2));

    return uv;
  } else {
    double theta = acos(p3d(2) / cv::norm(p3d));
    double phi = atan2(p3d(1), p3d(0));
    double k2 = D(0);
    double k3 = D(1);
    double k4 = D(2);
    double k5 = D(3);

    cv::Point2d p_u =
        r(k2, k3, k4, k5, theta) * cv::Point2d(cos(phi), sin(phi));

    // Apply generalised projection matrix
    cv::Point2d uv(K(0, 0) * p_u.x + K(0, 2), K(1, 1) * p_u.y + K(1, 2));
    return uv;
  }
}

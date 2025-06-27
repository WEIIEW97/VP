import cv2
import numpy as np
import math

class PoseGather:
    def __init__(self, pitch_offset=0.0):
        self.pitch_offset = pitch_offset

    def get_pose_imu(self, yaw, pitch, roll):
        ypr = np.zeros(3)
        ypr[0] = roll
        ypr[1] = -yaw
        ypr[2] = 90 - (pitch - self.pitch_offset)
        return ypr


def ypr2r(ypr: np.ndarray | list):
    y, p, r = np.deg2rad(ypr)
    Rz = [math.cos(y), -math.sin(y), 0, math.sin(y), math.cos(y), 0, 0, 0, 1]
    Ry = [math.cos(p), 0, math.sin(p), 0, 1, 0, -math.sin(p), 0, math.cos(p)]
    Rx = [1, 0, 0, 0, math.cos(r), -math.sin(r), 0, math.sin(r), math.cos(r)]

    Rz = np.array(Rz).reshape((3, 3))
    Ry = np.array(Ry).reshape((3, 3))
    Rx = np.array(Rx).reshape((3, 3))

    return Rz @ Ry @ Rx


def r2ypr(R: np.ndarray):
    n = R[:, 0]
    o = R[:, 1]
    a = R[:, 2]

    y = np.arctan2(n[1], n[0])
    p = np.arctan2(-n[2], n[0] * np.cos(y) + n[1] * np.sin(y))
    r = np.arctan2(
        a[0] * np.sin(y) - a[1] * np.cos(y), -o[0] * np.sin(y) + o[1] * np.cos(y)
    )
    ypr = np.array([y, p, r])
    return np.rad2deg(ypr)


class IPM:
    def __init__(
        self,
        K_src,
        dist,
        img_size,
        yaw_c_b,
        pitch_c_b,
        roll_c_b,
        tx_b_c,
        ty_b_c,
        tz_b_c,
        is_fisheye=False,
    ):
        self.K_src = K_src
        self.dist = dist
        self.img_size = img_size
        self.is_fisheye = is_fisheye

        if not self.is_fisheye:
            self.K_dst = cv2.getOptimalNewCameraMatrix(
                K_src, dist, img_size, -1, img_size, True
            )[0]
        else:
            self.K_dst = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K_src, dist[:4], img_size, np.eye(3)
            )

        R_c_b = self.YPR2R(np.array([yaw_c_b, pitch_c_b, roll_c_b]))
        t_b_c = np.array([tx_b_c, ty_b_c, tz_b_c])
        self.t_c_b = -R_c_b @ t_b_c

    def ProjectPointUV2BEVXY(
        self, uv, R_c_g=None, yaw_c_g=None, pitch_c_g=None, roll_c_g=None
    ):
        if R_c_g is None:
            R_c_g = self.YPR2R(np.array([yaw_c_g, pitch_c_g, roll_c_g]))

        uv_src = np.array([uv], dtype=np.float32)
        if not self.is_fisheye:
            uv_dst = cv2.undistortPoints(
                uv_src, self.K_src, self.dist, None, self.K_dst
            )
        else:
            uv_dst = cv2.fisheye.undistortPoints(
                uv_src, self.K_dst, self.dist[:4], None, self.K_dst
            )

        H_g_c = self.TransformImage2Ground(R_c_g, self.t_c_b)
        H_g_c = H_g_c @ np.linalg.inv(self.K_dst)

        point_uv = np.array([uv_dst[0][0][0], uv_dst[0][0][1], 1])
        point_3d = H_g_c @ point_uv
        point_3d /= point_3d[2]
        return np.array(point_3d[:2])

    def ProjectPointsUV2BEVXY(
        self, uvs, R_c_g=None, yaw_c_g=None, pitch_c_g=None, roll_c_g=None
    ):
        if R_c_g is None:
            R_c_g = self.YPR2R(np.array([yaw_c_g, pitch_c_g, roll_c_g]))

        uv_src = np.array(uvs, dtype=np.float32).reshape(-1, 1, 2)
        if not self.is_fisheye:
            uv_dst = cv2.undistortPoints(
                uv_src, self.K_src, self.dist, None, self.K_dst
            )
        else:
            uv_dst = cv2.fisheye.undistortPoints(
                uv_src, self.K_dst, self.dist[:4], None, self.K_dst
            )

        H_g_c = self.TransformImage2Ground(R_c_g, self.t_c_b)
        H_g_c = H_g_c @ np.linalg.inv(self.K_dst)

        result = []
        for item in uv_dst:
            point_uv = np.array([item[0][0], item[0][1], 1])
            point_3d = H_g_c @ point_uv
            point_3d /= point_3d[2]
            result.append(point_3d[:2])
        return np.array(result)

    def ProjectBEVXY2PointUV(
        self, xy, R_c_g=None, yaw_c_g=None, pitch_c_g=None, roll_c_g=None
    ):
        if R_c_g is None:
            R_c_g = self.YPR2R(np.array([yaw_c_g, pitch_c_g, roll_c_g]))

        line = self.LimitBEVLine(R_c_g, self.t_c_b)
        bev_xy = np.array([xy[0], self.LimitBEVy(line, xy), 1])
        H_i_g = self.TransformGround2Image(R_c_g, self.t_c_b)
        xyz_i = H_i_g @ bev_xy
        uv = self.SpaceToPlane(xyz_i, self.K_src, self.dist, self.is_fisheye)
        return np.array(uv)

    def ProjectBEVXYs2PointUVs(
        self, xys, R_c_g=None, yaw_c_g=None, pitch_c_g=None, roll_c_g=None
    ):
        if R_c_g is None:
            R_c_g = self.YPR2R(np.array([yaw_c_g, pitch_c_g, roll_c_g]))

        line = self.LimitBEVLine(R_c_g, self.t_c_b)
        H_i_g = self.TransformGround2Image(R_c_g, self.t_c_b)

        result = []
        for xy in xys:
            bev_xy = np.array([xy[0], self.LimitBEVy(line, xy), 1])
            xyz_i = H_i_g @ bev_xy
            uv = self.SpaceToPlane(xyz_i, self.K_src, self.dist, self.is_fisheye)
            result.append(uv)
        return np.array(result)

    def EstimateImuPitchOffset(
        self, marker_uv, marker_xy, yaw_c_g, pitch_c_g, roll_c_g, h_g_c
    ):
        uv_src = np.array([marker_uv], dtype=np.float32)
        if not self.is_fisheye:
            uv_dst = cv2.undistortPoints(
                uv_src, self.K_src, self.dist, None, self.K_dst
            )
        else:
            uv_dst = cv2.fisheye.undistortPoints(
                uv_src, self.K_dst, self.dist[:4], None, self.K_dst
            )

        beta = (
            math.atan2(uv_dst[0][0][1] - self.K_dst[1, 2], self.K_dst[1, 1])
            * 180
            / math.pi
        )
        alpha = math.atan2(marker_xy[1], h_g_c) * 180 / math.pi
        pitch_measurement = 180 - alpha - beta
        pitch_imu_est = roll_c_g
        return pitch_measurement - pitch_imu_est

    # def GetIPMImage(
    #     self, image, ipm_info, R_c_g=None, yaw_c_g=None, pitch_c_g=None, roll_c_g=None
    # ):
    #     if R_c_g is None:
    #         R_c_g = self.YPR2R(np.array([yaw_c_g, pitch_c_g, roll_c_g]))

    #     K_g = np.array(
    #         [
    #             [ipm_info["x_scale"], 0, 0.5 * (ipm_info["width"] - 1)],
    #             [0, -ipm_info["y_scale"], ipm_info["height"] - 1],
    #             [0, 0, 1],
    #         ]
    #     )
    #     ipm_size = (ipm_info["width"], ipm_info["height"])

    #     H_i_g = self.TransformGround2Image(R_c_g, self.t_c_b)
    #     H_i_g = self.K_dst @ H_i_g @ np.linalg.inv(K_g)

    #     if not self.is_fisheye:
    #         img_undist = cv2.undistort(image, self.K_src, self.dist, None, self.K_dst)
    #     else:
    #         img_undist = cv2.fisheye.undistortImage(
    #             image, self.K_src, self.dist[:4], None, self.K_dst
    #         )

    #     map_x = np.zeros(ipm_size[::-1], dtype=np.float32)
    #     map_y = np.zeros(ipm_size[::-1], dtype=np.float32)

    #     for i in range(ipm_size[1]):
    #         for j in range(ipm_size[0]):
    #             xy = np.array([j, i, 1])
    #             uv = H_i_g @ xy
    #             uv /= uv[2]
    #             map_x[i, j] = uv[0]
    #             map_y[i, j] = uv[1]

    #     ipm_img = cv2.remap(img_undist, map_x, map_y, cv2.INTER_LINEAR)
    #     return ipm_img

    def LimitBEVLine(self, R_c_g, t_c_g):
        bot_limit = [
            [0, self.img_size[1] * 1.5],
            [self.img_size[0] * 0.5, self.img_size[1] * 1.5],
            [self.img_size[0], self.img_size[1] * 1.5],
        ]
        bot_limit = self.ProjectPointsUV2BEVXY(bot_limit, R_c_g)

        a = np.array([bot_limit[0][0], bot_limit[0][1], 1])
        b = np.array([bot_limit[1][0], bot_limit[1][1], 1])
        c = np.array([bot_limit[2][0], bot_limit[2][1], 1])

        line1 = np.cross(a, b)
        line2 = np.cross(b, c)

        return [line1, line2, np.array([a[1], b[1], c[1]])]

    def LimitBEVy(self, line, xy):
        y = xy[1]
        limit_y1 = -(line[0][0] * xy[0] + line[0][2]) / line[0][1]
        limit_y2 = -(line[1][0] * xy[0] + line[1][2]) / line[1][1]

        limit_y = max(max(limit_y1, line[2][0]), max(limit_y2, line[2][2]))
        return y if y > limit_y else limit_y

    def TransformImage2Ground(self, R_c_g, t_c_g):
        return np.linalg.inv(self.TransformGround2Image(R_c_g, t_c_g))

    def TransformGround2Image(self, R_c_g, t_c_g):
        return np.array(
            [
                [R_c_g[0, 0], R_c_g[0, 1], t_c_g[0]],
                [R_c_g[1, 0], R_c_g[1, 1], t_c_g[1]],
                [R_c_g[2, 0], R_c_g[2, 1], t_c_g[2]],
            ]
        )

    def R2YPR(self, R):
        n = R[:, 0]
        o = R[:, 1]
        a = R[:, 2]

        y = math.atan2(n[1], n[0])
        p = math.atan2(-n[2], n[0] * math.cos(y) + n[1] * math.sin(y))
        r = math.atan2(
            a[0] * math.sin(y) - a[1] * math.cos(y),
            -o[0] * math.sin(y) + o[1] * math.cos(y),
        )

        return np.array([y, p, r]) * 180 / math.pi

    def YPR2R(self, ypr):
        y, p, r = ypr * math.pi / 180

        Rz = np.array(
            [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
        )

        Ry = np.array(
            [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
        )

        Rx = np.array(
            [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
        )

        return Rz @ Ry @ Rx

    def SpaceToPlane(self, p3d, K, D, is_fisheye):
        if not is_fisheye:
            p_u = (p3d[0] / p3d[2], p3d[1] / p3d[2])

            k1, k2, p1, p2, k3, k4, k5, k6 = D

            x, y = p_u
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            a1 = 2 * x * y
            a2 = r2 + 2 * x * x
            a3 = r2 + 2 * y * y
            cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
            icdist2 = 1.0 / (1 + k4 * r2 + k5 * r4 + k6 * r6)

            d_u = (
                x * cdist * icdist2 + p1 * a1 + p2 * a2 - x,
                y * cdist * icdist2 + p1 * a3 + p2 * a1 - y,
            )
            p_d = (p_u[0] + d_u[0], p_u[1] + d_u[1])

            return (K[0, 0] * p_d[0] + K[0, 2], K[1, 1] * p_d[1] + K[1, 2])
        else:
            k2, k3, k4, k5 = D[:4]
            theta = math.acos(p3d[2] / np.linalg.norm(p3d))
            phi = math.atan2(p3d[1], p3d[0])

            def r(k2, k3, k4, k5, theta):
                theta2 = theta * theta
                theta3 = theta2 * theta
                theta4 = theta2 * theta2
                theta5 = theta4 * theta
                theta6 = theta3 * theta3
                theta7 = theta6 * theta
                theta8 = theta4 * theta4
                theta9 = theta8 * theta
                return theta + k2 * theta3 + k3 * theta5 + k4 * theta7 + k5 * theta9

            r_val = r(k2, k3, k4, k5, theta)
            p_u = (r_val * math.cos(phi), r_val * math.sin(phi))
            return (K[0, 0] * p_u[0] + K[0, 2], K[1, 1] * p_u[1] + K[1, 2])


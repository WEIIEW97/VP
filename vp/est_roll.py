import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List


class CameraPoseSolver:
    """
    Optimized camera pose solver supporting both single-point and two-point modes
    with automatic switching between them.
    """
    def __init__(self, K: np.ndarray):
        """
        Initialize camera intrinsic parameters.

        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K

        # Cache recent results for smoothing
        self.last_roll = None
        self.last_R = None
        self.last_T = None

    @staticmethod
    def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Calculate rotation matrix from yaw, pitch, and roll angles."""
        # Convert inputs to numpy arrays and flatten
        yaw = np.asarray(yaw).ravel()[0]
        pitch = np.asarray(pitch).ravel()[0]
        roll = np.asarray(roll).ravel()[0]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        return R

    def solve_from_two_points(
        self,
        uv1: np.ndarray,
        uv2: np.ndarray,
        Pw1: np.ndarray,
        Pw2: np.ndarray,
        h: float,
        yaw: float,
        pitch: float,
    ) -> Dict:
        """
        Two-point pose estimation (more robust).

        Args:
            uv1, uv2: Two image points [u,v]
            Pw1, Pw2: Corresponding world points [x,y,z]
            h: Camera height
            yaw: Yaw angle (radians)
            pitch: Pitch angle (radians)

        Returns:
            Pose dictionary {'roll', 'R', 'T', 'reproj_error'}
        """
        # Convert to normalized coordinates
        # p1 = np.array([(uv1[0] - self.cx) / self.fx, (uv1[1] - self.cy) / self.fy, 1])
        # p2 = np.array([(uv2[0] - self.cx) / self.fx, (uv2[1] - self.cy) / self.fy, 1])

        def project(phi: float) -> Tuple[np.ndarray, np.ndarray]:
            """Project 3D points to image plane with given roll angle"""
            R = self.rotation_matrix(yaw, pitch, phi)
            T = np.array([0, 0, -h])
            Pc1 = R @ (Pw1 - T)
            Pc2 = R @ (Pw2 - T)
            uv1_reproj = (self.K @ Pc1) / Pc1[2]
            uv2_reproj = (self.K @ Pc2) / Pc2[2]
            return uv1_reproj[:2], uv2_reproj[:2]

        def objective(phi: float) -> np.ndarray:
            """Reprojection error objective function"""
            uv1_reproj, uv2_reproj = project(phi)
            return np.concatenate([uv1_reproj - uv1, uv2_reproj - uv2])

        # Try multiple initial guesses (including last result)
        init_guesses = np.linspace(-np.pi/6, np.pi/6, 7)
        if self.last_roll is not None:
            init_guesses = np.insert(init_guesses, 0, self.last_roll)

        best_solution = None
        min_error = float("inf")

        for guess in init_guesses:
            res = least_squares(objective, guess, method="lm")
            if res.success and res.cost < min_error:
                min_error = res.cost
                best_solution = res

        if best_solution is None:
            raise RuntimeError("Two-point optimization failed")

        roll = best_solution.x[0]
        R = self.rotation_matrix(yaw, pitch, roll)
        T = np.array([0, 0, -h])

        # Update cache
        self.last_roll = roll
        self.last_R = R
        self.last_T = T

        return {
            "roll": roll,
            "R": R,
            "T": T,
            "reproj_error": np.mean(np.abs(objective(roll))),
        }

    @staticmethod
    def safe_rad2deg(rad: float) -> float:
        """make sure the euler angle is in the range of [0, 180]"""
        deg = np.rad2deg(rad)
        if deg < 0:
            deg += 360
        return deg % 180

    def visualize(
        self,
        result: Dict,
        points_uv: List[np.ndarray] = None,
        points_3d: List[np.ndarray] = None,
    ):
        """
        Visualize the pose estimation results.

        Args:
            result: Pose dictionary from solve methods
            points_uv: Optional list of image points
            points_3d: Optional list of world points
        """
        fig = plt.figure(figsize=(12, 5))

        # 3D scene visualization
        ax1 = fig.add_subplot(121, projection="3d")
        if points_3d is not None:
            for Pw in points_3d:
                ax1.scatter(Pw[0], Pw[1], Pw[2], c="r", marker="o")

        T = result["T"]
        R = result["R"]
        ax1.scatter(T[0], T[1], T[2], c="b", marker="^", s=100)  # Camera position

        # Draw camera coordinate axes
        axis_length = 0.5
        colors = ["r", "g", "b"]  # RGB -> XYZ
        for i in range(3):
            ax1.quiver(
                T[0],
                T[1],
                T[2],
                R[0, i] * axis_length,
                R[1, i] * axis_length,
                R[2, i] * axis_length,
                color=colors[i],
                arrow_length_ratio=0.1,
            )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f'3D Pose (Roll: {np.rad2deg(result["roll"]):.1f}°)')

        # Reprojection visualization
        if points_uv is not None and points_3d is not None:
            ax2 = fig.add_subplot(122)
            for uv, Pw in zip(points_uv, points_3d):
                Pc = R @ (Pw - T)
                uv_reproj = (self.K @ Pc) / Pc[2]
                ax2.scatter(uv[0], uv[1], c="r")  # Original points
                ax2.scatter(
                    uv_reproj[0], uv_reproj[1], c="b", marker="x"
                )  # Reprojected

            ax2.set_xlabel("u")
            ax2.set_ylabel("v")
            ax2.set_title("Reprojection")
            ax2.legend(["Measured", "Reprojected"])

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example camera intrinsics
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # Create solver instance
    solver = CameraPoseSolver(K)

    # Simulated data
    h = 1.8  # Camera height
    yaw, pitch = np.deg2rad(10), np.deg2rad(-5)  # Known angles
    roll_gt = np.deg2rad(8)  # Ground truth roll to estimate

    # World points (now in [x,y,z] order)
    # Pw1 = np.array([2.0, 1.5, 0])  # Previously [2.0, 0, 1.5]
    # Pw2 = np.array([3.5, 2.0, 0])  # Previously [3.5, 0, 2.0]
    Pw1 = np.array([0, 5, 0])
    Pw2 = np.array([0, 3, 0])
    # Compute ground truth projections
    R_gt = solver.rotation_matrix(yaw, pitch, roll_gt)
    T_gt = np.array([0, 0, -h])  # Camera is at [0,0,h] looking down
    Pc1 = R_gt @ (Pw1 - T_gt)
    Pc2 = R_gt @ (Pw2 - T_gt)
    uv1 = (K @ Pc1) / Pc1[2]
    uv2 = (K @ Pc2) / Pc2[2]

    # Add noise to pixel coordinates
    uv1[:2] += np.random.normal(0, 1, 2)
    uv2[:2] += np.random.normal(0, 1, 2)

    print("=== Two-point solution ===")
    result = solver.solve_from_two_points(uv1[:2], uv2[:2], Pw1, Pw2, h, yaw, pitch)
    print(f"Roll: {solver.safe_rad2deg(result['roll']):.2f}° (GT: {np.rad2deg(roll_gt):.2f}°)")
    print(f"Reprojection error: {result['reproj_error']:.2f} pixels")
    # solver.visualize(result, [uv1[:2], uv2[:2]], [Pw1, Pw2])

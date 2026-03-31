"""
Task 2.1 — Inverse Kinematics for 5 end-effector poses.

Tests IK for each target pose, reports success/failure and the reason,
then sequentially commands the robot (sim or hw) to each feasible pose.

Run in simulation:
  ros2 launch lerobot sim_position.launch.py
  ros2 run python_controllers ik_poses

Run on hardware:
  ros2 launch lerobot hw_position.launch.py
  ros2 run python_controllers ik_poses
"""

import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .robot_kinematics import ik_robust, fk, HOME_Q, JOINT_NAMES, JOINT_LIMITS


# Target poses: (position [x,y,z], orientation [roll,pitch,yaw], label)
# Rotations are RPY in world frame: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
POSES = [
    ([0.2,   0.2,    0.2  ], [0.000,  1.570,  0.650], "I  "),
    ([0.2,   0.1,    0.4  ], [0.000,  0.000, -1.570], "II "),
    ([0.0,   0.0,    0.4  ], [0.000, -0.785,  1.570], "III"),
    ([0.0,   0.0,    0.07 ], [3.141,  0.000,  0.000], "IV "),
    ([0.0,   0.0452, 0.45 ], [-0.785, 0.000,  3.141], "V  "),
]

# Time to hold each pose [s]
DWELL_TIME = 3.0


class IKPosesNode(Node):
    def __init__(self):
        super().__init__('ik_poses')
        self._pub = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # Solve all poses first, then start moving
        self._feasible = []          # list of (label, q)
        self._move_index = -1        # -1 = move to home first
        self._solved = False

        # Solve on first tick, then start move loop
        self.create_timer(0.1, self._init_solve)

    # ------------------------------------------------------------------
    # IK solve phase
    # ------------------------------------------------------------------

    def _init_solve(self):
        """Run IK for all poses (once), then kick off the movement loop."""
        if self._solved:
            return
        self._solved = True

        self.get_logger().info('')
        self.get_logger().info('=== Task 2.1 — Inverse Kinematics ===')
        self.get_logger().info(f'{"Pose":<5} {"Status":<12} {"pos_err":>9} {"rot_err":>9}')
        self.get_logger().info('-' * 42)

        for p, rpy, label in POSES:
            q, ok, perr, rerr = ik_robust(
                np.array(p, dtype=float),
                np.array(rpy, dtype=float),
                n_tries=20,
            )

            status = 'FEASIBLE' if ok else 'INFEASIBLE'
            self.get_logger().info(
                f'{label:<5} {status:<12} {perr*1000:>7.1f}mm {np.degrees(rerr):>8.1f}°'
            )

            if ok:
                q_str = '  '.join(f'{np.degrees(qi):+.1f}°' for qi in q)
                self.get_logger().info(f'      q = [{q_str}]')
                self._feasible.append((label.strip(), q))
            else:
                reason = _infeasible_reason(np.array(p), np.array(rpy), q)
                self.get_logger().info(f'      Reason: {reason}')

        self.get_logger().info('')
        self.get_logger().info(
            f'Result: {len(self._feasible)}/{len(POSES)} poses feasible — '
            f'will now demonstrate on robot.'
        )

        # Start movement sequence: home -> each feasible pose
        self.create_timer(DWELL_TIME, self._step_movement)

    # ------------------------------------------------------------------
    # Movement phase
    # ------------------------------------------------------------------

    def _step_movement(self):
        if self._move_index == -1:
            self.get_logger().info('Moving to home position...')
            self._publish_q(HOME_Q)
        elif self._move_index < len(self._feasible):
            label, q = self._feasible[self._move_index]
            self.get_logger().info(f'Moving to pose {label}...')
            self._publish_q(q)
        else:
            self.get_logger().info('Done — all feasible poses demonstrated.')
            return

        self._move_index += 1

    def _publish_q(self, q):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(qi) for qi in q]
        msg.points = [pt]
        self._pub.publish(msg)


# ------------------------------------------------------------------
# Helper: explain why a pose is infeasible
# ------------------------------------------------------------------

def _infeasible_reason(p_des, rpy_des, q_best):
    """Return a short human-readable reason for IK failure."""
    T = fk(q_best)
    pos_err = np.linalg.norm(p_des - T[:3, 3])

    # Check if position alone is unreachable (position error >> 0)
    if pos_err > 0.05:
        return f'position likely outside workspace (residual {pos_err*1000:.0f} mm)'

    # Check if any joint is at its limit
    at_limit = []
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        if abs(q_best[i] - lo) < 0.01 or abs(q_best[i] - hi) < 0.01:
            at_limit.append(JOINT_NAMES[i])
    if at_limit:
        return f'joint limit reached: {", ".join(at_limit)}'

    return ('orientation requires >5 DoF or is at a kinematic singularity')


def main(args=None):
    rclpy.init(args=args)
    node = IKPosesNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
Task 2.3 — Circle trajectory using Jacobian IK.

Traces a circle with the gripper_center end-effector.
The circle lies in a HORIZONTAL plane (XY-plane) at the height of the
zero-joint-angle end-effector position (q = [0,0,0,0,0]).

  - Circle centre  : FK(q=0) = [0, 0.374, 0.237] m
  - Circle plane   : XY (z = 0.237 m, constant)
  - The circle is parametrised in time: one full revolution per PERIOD seconds.
  - Position-only Jacobian IK (3x5) is used for exact position tracking.
  - At each timestep the IK is warm-started from the previous joint solution
    to ensure a smooth, continuous joint trajectory.
  - A brief homing phase brings the arm to q=0 before the trajectory begins.

Run in simulation:
  ros2 launch lerobot sim_position.launch.py
  ros2 run python_controllers ik_circle

Run on hardware:
  ros2 launch lerobot hw_position.launch.py
  ros2 run python_controllers ik_circle

Parameters (edit the constants below):
  CIRCLE_RADIUS  : radius [m]
  PERIOD         : time for one full revolution [s]
  N_REVOLUTIONS  : how many times to trace the circle
  FREQ           : control-loop frequency [Hz]
"""

import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .robot_kinematics import fk, jacobian, JOINT_LIMITS, JOINT_NAMES

# ---------------------------------------------------------------------------
# Circle parameters — tune these to fit your workspace
# ---------------------------------------------------------------------------

CIRCLE_RADIUS  = 0.05     # [m]   radius of the circle
PERIOD         = 8.0      # [s]   time for one full revolution
N_REVOLUTIONS  = 3        # [-]   how many circles to trace
FREQ           = 25.0     # [Hz]  control-loop frequency

# IK settings (position-only DLS)
IK_LAM   = 0.005          # DLS damping
IK_ALPHA = 0.8            # step size
IK_ITER  = 300            # max iterations per step
IK_TOL   = 1e-4           # position convergence tolerance

# Zero-joint-angle configuration (circle centre reference)
Q_ZERO = np.zeros(5)


class CircleTrajectoryNode(Node):

    def __init__(self):
        super().__init__('ik_circle')
        self._pub = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # State machine: 'init' -> 'zero' -> 'circle' -> 'done'
        self._state = 'init'
        self._t_phase = 0.0        # elapsed time within current phase
        self._q_cur = Q_ZERO.copy()

        # Will be filled during init
        self._center = None        # circle centre [x, y, z]
        self._q_start = None       # joint angles at circle start point

        dt = 1.0 / FREQ
        self._timer = self.create_timer(dt, self._step)
        self.get_logger().info('Circle trajectory node started.')

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _step(self):
        dt = 1.0 / FREQ
        self._t_phase += dt

        if self._state == 'init':
            self._do_init()

        elif self._state == 'zero':
            self._do_zero()

        elif self._state == 'circle':
            self._do_circle()

        elif self._state == 'done':
            pass  # idle

    # ------------------------------------------------------------------
    # Phase: init — compute circle geometry from FK(q=0)
    # ------------------------------------------------------------------

    def _do_init(self):
        # Circle centre = EE position when all joints are at zero
        self._center = fk(Q_ZERO)[:3, 3].copy()

        self.get_logger().info(f'Circle centre (FK at q=0): {np.round(self._center, 3)} m')
        self.get_logger().info(f'Circle plane : XY  (z = {self._center[2]:.3f} m, constant)')
        self.get_logger().info(f'Circle radius: {CIRCLE_RADIUS} m')
        self.get_logger().info(f'Circle period: {PERIOD} s  ({N_REVOLUTIONS} revolution(s))')

        # Pre-compute IK for start point (theta = 0, rightmost +X point)
        p_start = self._circle_point(0.0)
        self._q_start = self._ik_pos(p_start, Q_ZERO, max_iter=1000, tol=1e-5)
        perr = float(np.linalg.norm(p_start - fk(self._q_start)[:3, 3]))
        if perr > 5e-3:
            self.get_logger().warn(
                f'IK for start point residual = {perr*1000:.1f} mm — '
                f'reduce radius if tracking is poor.'
            )
        self._q_cur = Q_ZERO.copy()

        self.get_logger().info('Moving to q=0 (zero position)...')
        self._t_phase = 0.0
        self._state = 'zero'

    # ------------------------------------------------------------------
    # Phase: zero — move to q=[0,0,0,0,0] over ~2 s
    # ------------------------------------------------------------------

    def _do_zero(self):
        self._publish_q(Q_ZERO)
        if self._t_phase >= 2.0:
            self.get_logger().info('Moving to circle start point...')
            self._q_cur = self._q_start.copy()
            self._publish_q(self._q_cur)
            self._t_phase = 0.0
            rclpy.spin_once(self, timeout_sec=0.0)
            self.create_timer(1.5, self._start_circle)

    def _start_circle(self):
        self.get_logger().info('Starting circle trajectory...')
        self._t_phase = 0.0
        self._state = 'circle'

    # ------------------------------------------------------------------
    # Phase: circle — trace the circle
    # ------------------------------------------------------------------

    def _do_circle(self):
        t = self._t_phase
        total = PERIOD * N_REVOLUTIONS

        if t > total:
            self.get_logger().info('Circle trajectory complete. Returning home...')
            self._state = 'done'
            self._publish_q(HOME_Q)
            return

        p_des = self._circle_point(t)

        # Position-only Jacobian IK, warm-started from previous solution
        q_new = self._ik_pos(p_des, self._q_cur)
        perr = float(np.linalg.norm(p_des - fk(q_new)[:3, 3]))

        if perr > 0.02:
            self.get_logger().warn(
                f't={t:.2f}s: IK residual large — pos={perr*1000:.1f}mm. '
                f'Point may be near workspace boundary.'
            )
        else:
            self._q_cur = q_new

        self._publish_q(self._q_cur)

        # Log progress once per revolution
        angle_deg = (t / PERIOD * 360.0) % 360.0
        rev = int(t / PERIOD) + 1
        if abs(angle_deg % 90) < (360.0 / PERIOD / FREQ + 1):
            self.get_logger().info(
                f'Rev {rev}/{N_REVOLUTIONS}  {angle_deg:5.0f}°  '
                f'p_des={np.round(p_des,3)}  pos_err={perr*1000:.1f}mm'
            )

    # ------------------------------------------------------------------
    # Circle geometry
    # ------------------------------------------------------------------

    def _circle_point(self, t: float) -> np.ndarray:
        """
        Position on the circle at time t [s].

        The circle lies in the XY plane (horizontal):
          x(t) = cx + R * cos(2π t / T)
          y(t) = cy + R * sin(2π t / T)
          z(t) = cz                        (constant = FK(q=0).z)

        theta=0 starts at the rightmost (+X) point.
        """
        theta = 2.0 * np.pi * t / PERIOD
        cx, cy, cz = self._center
        return np.array([
            cx + CIRCLE_RADIUS * np.cos(theta),
            cy + CIRCLE_RADIUS * np.sin(theta),
            cz,
        ])

    def _ik_pos(self, p_des, q_init, max_iter=None, tol=None):
        """Position-only Jacobian IK (3x5 DLS)."""
        if max_iter is None:
            max_iter = IK_ITER
        if tol is None:
            tol = IK_TOL
        q = np.array(q_init, dtype=float)
        for _ in range(max_iter):
            e = p_des - fk(q)[:3, 3]
            if np.linalg.norm(e) < tol:
                break
            J = jacobian(q)[:3, :]
            dq = IK_ALPHA * J.T @ np.linalg.solve(J @ J.T + IK_LAM**2 * np.eye(3), e)
            q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return q

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish_q(self, q):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(qi) for qi in q]
        msg.points = [pt]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CircleTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

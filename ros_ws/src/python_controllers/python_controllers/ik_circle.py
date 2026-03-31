"""
Task 2.3 — Circle trajectory using Jacobian IK.

Traces a circle with the gripper_center end-effector.
The circle lies in a VERTICAL plane in front of the robot.

  - The circle is parametrised in time: one full revolution per PERIOD seconds.
  - At each timestep the IK is warm-started from the previous joint solution
    to ensure a smooth, continuous joint trajectory.
  - A brief homing phase brings the arm to the circle's start point before
    the trajectory begins.

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

from .robot_kinematics import fk, ik, ik_robust, rot_to_rpy, HOME_Q, JOINT_NAMES

# ---------------------------------------------------------------------------
# Circle parameters — tune these to fit your workspace
# ---------------------------------------------------------------------------

CIRCLE_RADIUS  = 0.05     # [m]   radius of the circle
PERIOD         = 8.0      # [s]   time for one full revolution
N_REVOLUTIONS  = 3        # [-]   how many circles to trace
FREQ           = 25.0     # [Hz]  control-loop frequency

# IK settings
IK_LAM   = 0.03           # DLS damping
IK_ALPHA = 0.8            # step size (smaller = smoother)
IK_ITER  = 300            # max iterations per step
IK_TOL   = 2e-3           # position + rot tolerance


class CircleTrajectoryNode(Node):

    def __init__(self):
        super().__init__('ik_circle')
        self._pub = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # State machine: 'init' -> 'home' -> 'circle' -> 'done'
        self._state = 'init'
        self._t_phase = 0.0        # elapsed time within current phase
        self._q_cur = HOME_Q.copy()

        # Will be filled during init
        self._center = None        # circle centre [x, y, z]
        self._rpy_fixed = None     # constant orientation during circle
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

        elif self._state == 'home':
            self._do_home()

        elif self._state == 'circle':
            self._do_circle()

        elif self._state == 'done':
            pass  # idle

    # ------------------------------------------------------------------
    # Phase: init — compute circle geometry from FK(HOME_Q)
    # ------------------------------------------------------------------

    def _do_init(self):
        T_home = fk(HOME_Q)
        p_home = T_home[:3, 3]
        R_home = T_home[:3, :3]

        # Circle centre: same as home EE position
        self._center = p_home.copy()
        self._rpy_fixed = rot_to_rpy(R_home)

        # The circle lives in the plane spanned by world X and world Z,
        # centred at the home EE position.
        # theta=0  → rightmost point (home position)
        self.get_logger().info(
            f'Circle centre  : {np.round(self._center, 3)}'
        )
        self.get_logger().info(
            f'Circle radius  : {CIRCLE_RADIUS} m'
        )
        self.get_logger().info(
            f'Circle period  : {PERIOD} s  ({N_REVOLUTIONS} revolution(s))'
        )
        self.get_logger().info(
            f'Orientation (RPY): {np.round(np.degrees(self._rpy_fixed), 1)} deg (fixed)'
        )

        # Pre-compute IK for the start point (theta = 0)
        p_start = self._circle_point(0.0)
        q_start, ok, perr, _ = ik_robust(
            p_start, self._rpy_fixed,
            n_tries=20,
            max_iter=IK_ITER, tol=IK_TOL, lam=IK_LAM, alpha=IK_ALPHA,
        )
        if not ok:
            self.get_logger().warn(
                f'IK for start point did not fully converge (pos_err={perr*1000:.1f} mm). '
                f'Proceeding anyway — reduce radius if tracking is poor.'
            )
        self._q_start = q_start
        self._q_cur = HOME_Q.copy()

        self.get_logger().info('Moving to home...')
        self._t_phase = 0.0
        self._state = 'home'

    # ------------------------------------------------------------------
    # Phase: home — move to HOME_Q over ~2 s
    # ------------------------------------------------------------------

    def _do_home(self):
        self._publish_q(HOME_Q)
        if self._t_phase >= 2.0:
            self.get_logger().info('Moving to circle start...')
            self._q_cur = self._q_start.copy()
            self._publish_q(self._q_cur)
            self._t_phase = 0.0
            # Wait a moment for the arm to reach the start
            rclpy.spin_once(self, timeout_sec=0.0)
            # Transition after an extra 1.5 s settle time
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

        # Jacobian IK warm-started from previous joint solution
        q_new, ok, perr, rerr = ik(
            p_des, self._rpy_fixed,
            q_init=self._q_cur,
            max_iter=IK_ITER,
            tol=IK_TOL,
            lam=IK_LAM,
            alpha=IK_ALPHA,
        )

        if not ok and perr > 0.02:
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

        The circle lies in the XZ plane:
          x(t) = cx + R * cos(2π t / T)
          y(t) = cy                        (constant)
          z(t) = cz + R * sin(2π t / T)

        theta=0 starts at the rightmost (+X) point.
        """
        theta = 2.0 * np.pi * t / PERIOD
        cx, cy, cz = self._center
        return np.array([
            cx + CIRCLE_RADIUS * np.cos(theta),
            cy,
            cz + CIRCLE_RADIUS * np.sin(theta),
        ])

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

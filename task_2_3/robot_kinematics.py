"""
Forward kinematics, geometric Jacobian, and Jacobian-based IK
for the LeRobot 5-DoF manipulator (RRRRR).

Frame chain (all revolute joints rotate about local Z-axis):
  world -> base (fixed, Rotz(π))
  base -> shoulder     (Shoulder_Rotation,  q1)
  shoulder -> upper_arm (Shoulder_Pitch,     q2)
  upper_arm -> lower_arm (Elbow,             q3)
  lower_arm -> wrist    (Wrist_Pitch,        q4)
  wrist -> gripper      (Wrist_Roll,         q5)
  gripper -> gripper_center (fixed)
"""

import numpy as np

JOINT_NAMES = [
    'Shoulder_Rotation',
    'Shoulder_Pitch',
    'Elbow',
    'Wrist_Pitch',
    'Wrist_Roll',
]

JOINT_LIMITS = np.array([
    [-2.0,      2.0     ],  # Shoulder_Rotation
    [-1.57,     1.57    ],  # Shoulder_Pitch
    [-1.58,     1.58    ],  # Elbow
    [-1.57,     1.57    ],  # Wrist_Pitch
    [-np.pi,    np.pi   ],  # Wrist_Roll
])

# Home position used by the hardware/sim nodes
HOME_Q = np.array([0.0, np.deg2rad(105), np.deg2rad(-70), np.deg2rad(-60), 0.0])


# ---------------------------------------------------------------------------
# Low-level transform helpers
# ---------------------------------------------------------------------------

def _rz(q: float) -> np.ndarray:
    """4x4 homogeneous rotation matrix about Z by angle q [rad]."""
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=float)


def _tf(xyz, rpy) -> np.ndarray:
    """
    4x4 homogeneous transform from translation xyz and RPY angles.
    Rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    rx, ry, rz_ = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz_), np.sin(rz_)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# ---------------------------------------------------------------------------
# URDF static offsets
# ---------------------------------------------------------------------------

_T_WB = _tf([0, 0, 0],            [0, 0, np.pi])       # world->base (fixed)
_OFF = [
    _tf([0, -0.0452, 0.0165],  [0, 0, 0]),              # base->shoulder
    _tf([0, -0.0306, 0.1025],  [0, -np.pi/2, 0]),       # shoulder->upper_arm
    _tf([0.11257, -0.028, 0],  [0, 0, 0]),               # upper_arm->lower_arm
    _tf([0.0052, -0.1349, 0],  [0, 0, np.pi/2]),         # lower_arm->wrist
    _tf([-0.0601, 0, 0],       [0, -np.pi/2, 0]),        # wrist->gripper
]
_T_GC = _tf([0, 0, 0.075], [0, 0, 0])                   # gripper->gripper_center (fixed)


# ---------------------------------------------------------------------------
# Public kinematics functions
# ---------------------------------------------------------------------------

def fk(q) -> np.ndarray:
    """
    Forward kinematics: 4x4 world->gripper_center transform.

    Args:
        q: array-like, 5 joint angles [rad]

    Returns:
        T: (4,4) homogeneous transform
    """
    q = np.asarray(q, dtype=float)
    T = _T_WB.copy()
    for i in range(5):
        T = T @ _OFF[i] @ _rz(q[i])
    return T @ _T_GC


def fk_link_positions(q) -> np.ndarray:
    """
    Return (7, 3) array of joint positions in world frame for visualisation.
    Points: world/base, shoulder, upper_arm, lower_arm, wrist, gripper, gripper_center.
    """
    q = np.asarray(q, dtype=float)
    positions = [_T_WB[:3, 3].copy()]
    T = _T_WB.copy()
    for i in range(5):
        T = T @ _OFF[i] @ _rz(q[i])
        positions.append(T[:3, 3].copy())
    positions.append((T @ _T_GC)[:3, 3].copy())
    return np.array(positions)


def jacobian(q) -> np.ndarray:
    """
    6x5 geometric Jacobian at joint angles q.

    Column i for revolute joint i rotating about z_i:
        J_v_i = z_i x (p_ee - p_i)    (linear velocity)
        J_w_i = z_i                     (angular velocity)

    Args:
        q: array-like, 5 joint angles [rad]

    Returns:
        J: (6, 5) Jacobian matrix
    """
    q = np.asarray(q, dtype=float)
    joints = []
    T = _T_WB.copy()
    for i in range(5):
        T = T @ _OFF[i]
        joints.append((T[:3, 3].copy(), T[:3, 2].copy()))  # (p_i, z_i)
        T = T @ _rz(q[i])
    pe = (T @ _T_GC)[:3, 3]

    J = np.zeros((6, 5))
    for i, (pi, zi) in enumerate(joints):
        J[:3, i] = np.cross(zi, pe - pi)
        J[3:, i] = zi
    return J


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def rpy_to_rot(rpy) -> np.ndarray:
    """RPY (roll, pitch, yaw) -> 3x3 rotation matrix. R = Rz(yaw)@Ry(pitch)@Rx(roll)."""
    return _tf([0, 0, 0], rpy)[:3, :3]


def rot_to_rpy(R) -> np.ndarray:
    """3x3 rotation matrix -> RPY (roll, pitch, yaw)."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if abs(abs(pitch) - np.pi/2) < 1e-6:   # gimbal lock
        yaw = np.arctan2(R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw  = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])


def _rot_error(R_des, R_cur) -> np.ndarray:
    """Axis-angle rotation error vector: theta * axis."""
    Re = R_des @ R_cur.T
    cos_a = np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_a)
    if theta < 1e-8:
        return np.zeros(3)
    return (theta / (2 * np.sin(theta))) * np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1],
    ])


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

def ik(p_des, rpy_des, q_init=None, max_iter=500, tol=5e-4,
       lam=0.05, alpha=1.0, rot_scale=0.3):
    """
    Jacobian-based IK via damped least squares (DLS).

    Finds q such that FK(q) matches the desired position and orientation.
    The robot has 5 DoF; a 6D task may be underdetermined — DLS gives the
    least-norm joint motion.

    Args:
        p_des     : desired position [x, y, z] in world frame
        rpy_des   : desired orientation [roll, pitch, yaw] in world frame
        q_init    : initial joint angles (default: HOME_Q)
        max_iter  : maximum iteration count
        tol       : convergence tolerance (pos_err + rot_scale*rot_err)
        lam       : DLS damping factor (larger = more stable near singularities)
        alpha     : step size (0 < alpha <= 1)
        rot_scale : weight for rotation rows relative to position rows

    Returns:
        q       : (5,) joint angles [rad], clipped to joint limits
        success : True if converged within tolerances
        pos_err : final position error [m]
        rot_err : final rotation error [rad]
    """
    if q_init is None:
        q_init = HOME_Q.copy()
    q = np.asarray(q_init, dtype=float).copy()
    R_des = rpy_to_rot(rpy_des)

    for _ in range(max_iter):
        T = fk(q)
        e_p = p_des - T[:3, 3]
        e_r = _rot_error(R_des, T[:3, :3])

        if np.linalg.norm(e_p) + rot_scale * np.linalg.norm(e_r) < tol:
            break

        J = jacobian(q)
        J[3:] *= rot_scale
        e = np.concatenate([e_p, rot_scale * e_r])

        # DLS: dq = alpha * J^T (J J^T + lam^2 I)^{-1} e
        dq = alpha * J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T = fk(q)
    pos_err = float(np.linalg.norm(p_des - T[:3, 3]))
    rot_err = float(np.linalg.norm(_rot_error(R_des, T[:3, :3])))
    success = pos_err < 1e-2 and rot_err < 0.15
    return q, success, pos_err, rot_err


def ik_robust(p_des, rpy_des, n_tries=12, seed=42, **ik_kwargs):
    """
    Run IK from multiple starting configurations; return the best solution.

    Args:
        p_des, rpy_des: target pose
        n_tries       : number of random restarts (+ home init)
        seed          : RNG seed for reproducibility
        **ik_kwargs   : forwarded to ik()

    Returns:
        q, success, pos_err, rot_err  (same as ik())
    """
    rng = np.random.default_rng(seed)
    inits = [HOME_Q.copy()]
    for _ in range(n_tries - 1):
        inits.append(rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]))

    best = None
    best_score = float('inf')
    for q0 in inits:
        result = ik(np.asarray(p_des), np.asarray(rpy_des), q_init=q0, **ik_kwargs)
        score = result[2] + 0.3 * result[3]
        if score < best_score:
            best_score = score
            best = result
    return best


def find_multiple_solutions(p_des, rpy_des, n_samples=100, angle_tol=0.3, seed=0, **ik_kwargs):
    """
    Find multiple distinct IK solutions by sampling random starting configs.

    Returns:
        solutions: list of (q,) arrays for each unique feasible solution
    """
    rng = np.random.default_rng(seed)
    solutions = []

    for _ in range(n_samples):
        q0 = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        q, ok, _, _ = ik(np.asarray(p_des), np.asarray(rpy_des), q_init=q0, **ik_kwargs)
        if not ok:
            continue
        # Check if this solution is distinct from already-found ones
        is_new = True
        for q_existing in solutions:
            if np.linalg.norm(q - q_existing) < angle_tol:
                is_new = False
                break
        if is_new:
            solutions.append(q.copy())

    return solutions

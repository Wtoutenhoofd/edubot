"""
Task 2.3 — Standalone circle trajectory simulation (no ROS2 required).

Simulates the gripper tracing a horizontal circle (XY-plane) centred at
the end-effector position for q = [0,0,0,0,0], and plots the result.

Usage:
    python3 simulate_circle.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from robot_kinematics import fk, jacobian, fk_link_positions, JOINT_NAMES, JOINT_LIMITS

# ---------------------------------------------------------------------------
# Trajectory parameters — edit these to change the circle
# ---------------------------------------------------------------------------
RADIUS = 0.05   # [m]   circle radius
PERIOD = 8.0    # [s]   time for one full revolution
N_REVS = 2      # [-]   number of revolutions to simulate
FREQ   = 25.0   # [Hz]  control-loop frequency (timesteps per second)

# IK solver parameters (damped least squares)
IK_LAM   = 0.005  # damping factor λ — prevents large steps near singularities
IK_ALPHA = 0.8    # step size α — scales the joint update (0 < α ≤ 1)
IK_ITER  = 300    # maximum iterations per timestep
IK_TOL   = 1e-4   # convergence tolerance [m]


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

def ik_pos(p_des: np.ndarray, q_init: np.ndarray,
           max_iter: int = IK_ITER, tol: float = IK_TOL,
           lam: float = IK_LAM, alpha: float = IK_ALPHA) -> np.ndarray:
    """
    Position-only Jacobian IK via damped least squares (DLS).

    Uses only the top 3 rows of the 6x5 Jacobian (linear velocity part),
    giving a 3x5 system. With 5 joints and 3 constraints the null space has
    dimension 2, so the DLS solution finds the least-norm joint motion:

        dq = α · J_pos^T (J_pos · J_pos^T + λ²I)^{-1} · e_pos

    The IK should be warm-started from the previous joint solution to ensure
    a smooth, continuous trajectory without jumps.

    Args:
        p_des    : desired end-effector position [x, y, z] in world frame
        q_init   : starting joint configuration for the iteration
        max_iter : maximum number of iterations
        tol      : convergence tolerance on position error [m]
        lam      : DLS damping factor λ
        alpha    : step size α

    Returns:
        q : (5,) joint angles [rad], clipped to joint limits
    """
    q = q_init.copy()
    for _ in range(max_iter):
        e = p_des - fk(q)[:3, 3]           # position error vector (3,)
        if np.linalg.norm(e) < tol:
            break
        J = jacobian(q)[:3, :]              # position rows of Jacobian (3x5)
        # DLS: dq = α · J^T (J J^T + λ²I)^{-1} · e
        dq = alpha * J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), e)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def circle_point(center: np.ndarray, t: float) -> np.ndarray:
    """
    Return the target end-effector position on the circle at time t [s].

    The circle lies in the horizontal XY-plane:
        x(t) = cx + R·cos(2π t / T)
        y(t) = cy + R·sin(2π t / T)
        z(t) = cz   (constant)
    """
    theta = 2 * np.pi * t / PERIOD
    return center + np.array([RADIUS * np.cos(theta), RADIUS * np.sin(theta), 0.0])


def simulate(center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the circle trajectory simulation.

    Args:
        center : circle centre position [x, y, z] in world frame

    Returns:
        times        : (N,)   time array [s]
        p_ideal      : (N, 3) ideal (target) end-effector positions
        ee_positions : (N, 3) achieved end-effector positions
        joint_traj   : (N, 5) joint angles at each timestep [rad]
    """
    times   = np.arange(0, PERIOD * N_REVS, 1.0 / FREQ)
    p_ideal = np.array([circle_point(center, t) for t in times])

    # Warm-start: bring IK to the circle start point before the trajectory loop
    q_cur = ik_pos(circle_point(center, 0.0), np.zeros(5), max_iter=1000, tol=1e-5)

    ee_positions = np.empty((len(times), 3))
    joint_traj   = np.empty((len(times), 5))

    for i, t in enumerate(times):
        # Solve IK warm-started from the previous timestep's joint solution
        q_cur = ik_pos(p_ideal[i], q_cur)
        ee_positions[i] = fk(q_cur)[:3, 3]
        joint_traj[i]   = q_cur

    return times, p_ideal, ee_positions, joint_traj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(center, times, p_ideal, ee_positions, joint_traj, save_path=None):
    """
    Produce a three-panel figure:
      1. 3D view with robot arm snapshots
      2. XY top-down view comparing target vs. achieved path
      3. Joint angles over time
    """
    errors = np.linalg.norm(ee_positions - p_ideal, axis=1)
    th     = np.linspace(0, 2 * np.pi, 360)   # smooth circle for plotting

    fig = plt.figure(figsize=(16, 5), facecolor='white')

    # --- Panel 1: 3D view ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('white')
    # Ideal circle
    ax1.plot(center[0] + RADIUS * np.cos(th),
             center[1] + RADIUS * np.sin(th),
             np.full_like(th, center[2]),
             'r--', lw=2.5, label='Target', zorder=5)
    # Achieved EE path
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
             color='steelblue', lw=2, label='EE pad')
    # Robot arm snapshots (8 equally spaced in time)
    snap_indices = np.linspace(0, len(times) - 1, 8, dtype=int)
    for k, idx in enumerate(snap_indices):
        pts = fk_link_positions(joint_traj[idx])
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                 color=plt.colormaps['plasma'](k / len(snap_indices)),
                 alpha=0.6, lw=1.5, ms=3)
    ax1.scatter(*center, s=80, color='red', zorder=10)
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
    ax1.set_title('3D — robot snapshots', fontsize=10)
    ax1.legend(fontsize=8)

    # --- Panel 2: XY top-down view ---
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor('white')
    ax2.plot(center[0] + RADIUS * np.cos(th),
             center[1] + RADIUS * np.sin(th),
             'r--', lw=2.5, label='Target')
    ax2.plot(ee_positions[:, 0], ee_positions[:, 1],
             color='steelblue', lw=2, label='Bereikt')
    ax2.scatter(center[0], center[1], s=80, color='red', zorder=5, label='Centrum')
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title(f'XY-vlak — bovenaanzicht (z = {center[2]:.3f} m)', fontsize=10)
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True)

    # --- Panel 3: Joint angles over time ---
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor('white')
    colors = ['#e74c3c', '#e67e22', '#2980b9', '#8e44ad', '#27ae60']
    for i, (name, color) in enumerate(zip(JOINT_NAMES, colors)):
        ax3.plot(times, np.degrees(joint_traj[:, i]),
                 color=color, lw=1.8, label=name.replace('_', ' '))
    ax3.set_xlabel('Tijd [s]'); ax3.set_ylabel('Hoek [deg]')
    ax3.set_title('Jointhoeken', fontsize=10)
    ax3.legend(fontsize=7); ax3.grid(True)

    fig.suptitle(
        f'Task 2.3 — Cirkel-traject in XY-vlak  '
        f'(R={RADIUS} m,  z={center[2]:.3f} m,  T={PERIOD} s,  {N_REVS}×)\n'
        f'Jacobian position-only IK  |  '
        f'gem. fout = {errors.mean()*1000:.2f} mm   '
        f'max fout = {errors.max()*1000:.2f} mm',
        fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    Q_ZERO = np.zeros(5)

    # Circle centre = end-effector position when all joints are at zero
    center = fk(Q_ZERO)[:3, 3]
    print(f"Circle centre (FK at q=0): {np.round(center, 4)} m")
    print(f"Circle plane : XY  (z = {center[2]:.3f} m, constant)")
    print(f"Radius: {RADIUS} m   Period: {PERIOD} s   Revolutions: {N_REVS}")
    print("Running simulation...")

    times, p_ideal, ee_positions, joint_traj = simulate(center)

    errors = np.linalg.norm(ee_positions - p_ideal, axis=1)
    print(f"Mean tracking error : {errors.mean()*1000:.2f} mm")
    print(f"Max  tracking error : {errors.max()*1000:.2f} mm")

    save_path = os.path.join(os.path.dirname(__file__), 'circle_simulation.png')
    plot(center, times, p_ideal, ee_positions, joint_traj, save_path=save_path)


if __name__ == '__main__':
    main()

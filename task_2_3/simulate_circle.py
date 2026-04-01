"""
Task 2.3 — Standalone circle trajectory simulation (no ROS2 required).

Simulates the gripper tracing a horizontal circle (XY-plane) centred at
the end-effector position for q = [0,0,0,0,0], and plots the result.

Usage:
    python3 simulate_circle.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from robot_kinematics import fk, jacobian, fk_link_positions, JOINT_NAMES, JOINT_LIMITS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS   = 0.05    # [m]
PERIOD   = 8.0     # [s]
N_REVS   = 2       # [-]
FREQ     = 25.0    # [Hz]
IK_LAM   = 0.005
IK_ALPHA = 0.8


def ik_pos(p_des, q_init, max_iter=300, tol=1e-4):
    """Position-only Jacobian IK via damped least squares (3x5)."""
    q = q_init.copy()
    for _ in range(max_iter):
        e = p_des - fk(q)[:3, 3]
        if np.linalg.norm(e) < tol:
            break
        J = jacobian(q)[:3, :]
        dq = IK_ALPHA * J.T @ np.linalg.solve(J @ J.T + IK_LAM**2 * np.eye(3), e)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q


def main():
    Q_ZERO = np.zeros(5)
    center = fk(Q_ZERO)[:3, 3]
    print(f"Circle centre (FK at q=0): {np.round(center, 4)} m")
    print(f"Circle plane : XY  (z = {center[2]:.3f} m, constant)")
    print(f"Radius       : {RADIUS} m,  Period: {PERIOD} s,  Revolutions: {N_REVS}")

    times = np.arange(0, PERIOD * N_REVS, 1.0 / FREQ)

    # Warm-start IK to circle start point
    q_cur = ik_pos(center + np.array([RADIUS, 0, 0]), Q_ZERO, max_iter=1000, tol=1e-5)

    ee_positions, joint_traj = [], []
    for t in times:
        theta = 2 * np.pi * t / PERIOD
        p_des = center + np.array([RADIUS * np.cos(theta), RADIUS * np.sin(theta), 0])
        q_cur = ik_pos(p_des, q_cur)
        ee_positions.append(fk(q_cur)[:3, 3])
        joint_traj.append(q_cur.copy())

    ee_positions = np.array(ee_positions)
    joint_traj   = np.array(joint_traj)

    p_ideal = np.array([
        center + np.array([RADIUS * np.cos(2 * np.pi * t / PERIOD),
                           RADIUS * np.sin(2 * np.pi * t / PERIOD), 0])
        for t in times
    ])
    errors = np.linalg.norm(ee_positions - p_ideal, axis=1)
    print(f"Mean tracking error: {errors.mean()*1000:.2f} mm")
    print(f"Max  tracking error: {errors.max()*1000:.2f} mm")

    # ---- Plot ----
    th = np.linspace(0, 2 * np.pi, 300)
    fig = plt.figure(figsize=(16, 5), facecolor='white')

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('white')
    ax1.plot(center[0] + RADIUS * np.cos(th), center[1] + RADIUS * np.sin(th),
             [center[2]] * 300, 'r--', lw=2.5, label='Target', zorder=5)
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
             color='steelblue', lw=2, label='EE pad')
    for k, idx in enumerate(np.linspace(0, len(times) - 1, 8, dtype=int)):
        pts = fk_link_positions(joint_traj[idx])
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                 color=plt.colormaps['plasma'](k / 8), alpha=0.6, lw=1.5, ms=3)
    ax1.scatter(*center, s=80, color='red', zorder=10)
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
    ax1.set_title('3D — robot snapshots', fontsize=10)
    ax1.legend(fontsize=8)

    # XY plane (top view)
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor('white')
    ax2.plot(center[0] + RADIUS * np.cos(th), center[1] + RADIUS * np.sin(th),
             'r--', lw=2.5, label='Target')
    ax2.plot(ee_positions[:, 0], ee_positions[:, 1], color='steelblue', lw=2, label='Bereikt')
    ax2.scatter(center[0], center[1], s=80, color='red', zorder=5)
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title(f'XY-vlak — bovenaanzicht (z = {center[2]:.3f} m)', fontsize=10)
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True)

    # Joint angles
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor('white')
    colors = ['#e74c3c', '#e67e22', '#2980b9', '#8e44ad', '#27ae60']
    for i, (name, c) in enumerate(zip(JOINT_NAMES, colors)):
        ax3.plot(times, np.degrees(joint_traj[:, i]), color=c, lw=1.8,
                 label=name.replace('_', ' '))
    ax3.set_xlabel('Tijd [s]'); ax3.set_ylabel('Hoek [deg]')
    ax3.set_title('Jointhoeken', fontsize=10)
    ax3.legend(fontsize=7); ax3.grid(True)

    fig.suptitle(
        f'Task 2.3 — Cirkel-traject in XY-vlak  '
        f'(R={RADIUS}m, z={center[2]:.3f}m, T={PERIOD}s, {N_REVS}×)\n'
        f'Jacobian position-only IK  |  '
        f'gem. fout = {errors.mean()*1000:.2f} mm  '
        f'max = {errors.max()*1000:.2f} mm',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'circle_simulation.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Figure saved: circle_simulation.png")
    plt.show()


if __name__ == '__main__':
    main()

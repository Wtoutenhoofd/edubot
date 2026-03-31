"""
Task 2.2 — Visualise multiple IK solutions for feasible poses.

Finds multiple distinct IK solutions by sampling random starting configs
and plots the robot arm (as a stick figure) for each solution.

This is a standalone matplotlib script — no ROS2 required.

Usage:
  python3 ik_solutions_viz.py

The script will print each unique solution and show a 3D matplotlib figure
with the robot drawn for every solution in a different colour.

At the bottom, it discusses which solutions are practically feasible on the
physical robot (joint-limit and singularity analysis).
"""

import sys
import os

# Allow running as a plain Python script from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')           # change to 'Agg' if no display available
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401

from robot_kinematics import (
    fk, fk_link_positions, find_multiple_solutions, rot_to_rpy,
    JOINT_NAMES, JOINT_LIMITS, HOME_Q,
)

# ---------------------------------------------------------------------------
# Feasible poses from Task 2.1  (fill in after running ik_poses)
# Only include the ones that converged in Task 2.1
# ---------------------------------------------------------------------------
FEASIBLE_POSES = [
    # label, position [x,y,z], orientation [roll,pitch,yaw]
    ("I",   [0.2,   0.2,   0.2 ], [0.000,  1.570,  0.650]),
    ("II",  [0.2,   0.1,   0.4 ], [0.000,  0.000, -1.570]),
    ("III", [0.0,   0.0,   0.4 ], [0.000, -0.785,  1.570]),
    ("IV",  [0.0,   0.0,   0.07], [3.141,  0.000,  0.000]),
    ("V",   [0.0,   0.0452, 0.45], [-0.785, 0.000,  3.141]),
]

N_SAMPLES   = 200    # random restarts per pose
ANGLE_TOL   = 0.30   # rad — two solutions are "distinct" if ||dq|| > this


def _draw_robot(ax, q, color, alpha=0.8, label=None):
    """Draw a stick-figure robot arm on Axes3D ax."""
    pts = fk_link_positions(q)   # (7, 3)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            '-o', color=color, alpha=alpha, linewidth=2,
            markersize=4, label=label)
    # Mark end-effector
    ax.scatter(*pts[-1], s=60, color=color, alpha=alpha, zorder=5)


def _joint_health(q):
    """
    Return a short string describing practical feasibility on the real robot.
    Checks joint-limit margin and rough singularity proximity.
    """
    warnings = []
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        margin = min(q[i] - lo, hi - q[i])
        if margin < 0.1:
            warnings.append(f'{JOINT_NAMES[i]} near limit ({np.degrees(q[i]):.0f}°)')

    # Rough singularity check: elbow or wrist near 0
    if abs(q[2]) < 0.15:
        warnings.append('Elbow near singularity (q3 ≈ 0)')
    if abs(q[3]) < 0.15:
        warnings.append('Wrist_Pitch near singularity (q4 ≈ 0)')

    if warnings:
        return 'CAUTION: ' + '; '.join(warnings)
    return 'OK'


def analyse_pose(label, p_des, rpy_des):
    """Find all IK solutions for one pose and draw them."""
    print(f'\n{"="*60}')
    print(f'Pose {label}: p={np.round(p_des,3)}, rpy={np.round(rpy_des,3)}')
    print(f'Searching for multiple solutions ({N_SAMPLES} random starts)...')

    solutions = find_multiple_solutions(
        p_des, rpy_des,
        n_samples=N_SAMPLES,
        angle_tol=ANGLE_TOL,
    )

    if not solutions:
        print('  No feasible solution found — pose is outside workspace.')
        return

    print(f'  Found {len(solutions)} distinct solution(s):')

    fig = plt.figure(figsize=(6 * len(solutions), 5))
    fig.suptitle(f'Pose {label} — {len(solutions)} IK solution(s)', fontsize=12)

    cmap = plt.colormaps['tab10']

    for idx, q in enumerate(solutions):
        T = fk(q)
        rpy_achieved = rot_to_rpy(T[:3, :3])

        print(f'\n  Solution {idx+1}:')
        for i, (name, qi) in enumerate(zip(JOINT_NAMES, q)):
            print(f'    {name:20s}: {np.degrees(qi):+7.1f}°')
        print(f'    Achieved position : {np.round(T[:3,3], 4)}')
        print(f'    Achieved RPY [deg]: {np.round(np.degrees(rpy_achieved), 1)}')
        print(f'    Physical robot    : {_joint_health(q)}')

        ax = fig.add_subplot(1, len(solutions), idx + 1, projection='3d')
        ax.set_title(f'Solution {idx+1}', fontsize=10)
        _draw_robot(ax, q, color=cmap(idx % 10))

        # Draw target position
        ax.scatter(*p_des, s=80, marker='*', color='red', zorder=10,
                   label='Target')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([0.0, 0.5])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'ik_solutions_pose_{label}.png', dpi=120)
    print(f'\n  Figure saved: ik_solutions_pose_{label}.png')
    plt.show(block=False)


def analyse_all_on_one_figure(poses_solutions):
    """Draw all solutions for all poses on a single multi-panel figure."""
    fig = plt.figure(figsize=(14, 4 * len(poses_solutions)))
    fig.suptitle('Multiple IK solutions per pose', fontsize=13)
    cmap = plt.colormaps['Set1']

    row = 0
    total_cols = max((len(s) for _, s in poses_solutions), default=1)
    total_cols = max(total_cols, 1)

    for label, solutions in poses_solutions:
        for col, q in enumerate(solutions):
            ax_idx = row * total_cols + col + 1
            ax = fig.add_subplot(
                len(poses_solutions), total_cols, ax_idx, projection='3d')
            ax.set_title(f'Pose {label} — sol {col+1}', fontsize=9)
            _draw_robot(ax, q, color=cmap(col % 9))
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([0.0, 0.5])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        row += 1

    plt.tight_layout()
    plt.savefig('ik_all_solutions.png', dpi=120)
    print('\nCombined figure saved: ik_all_solutions.png')
    plt.show()


def main():
    print('Task 2.2 — Multiple IK Solutions')
    print('==================================')
    print(f'Robot has 5 revolute joints → up to multiple local minima per pose.')
    print(f'Solutions are found by random-restart Jacobian IK ({N_SAMPLES} starts/pose).')

    all_results = []

    for label, p, rpy in FEASIBLE_POSES:
        analyse_pose(label, np.array(p), np.array(rpy))
        sols = find_multiple_solutions(
            np.array(p), np.array(rpy),
            n_samples=N_SAMPLES,
            angle_tol=ANGLE_TOL,
        )
        all_results.append((label, sols))

    analyse_all_on_one_figure(all_results)

    print('\n--- Practical feasibility on the physical robot ---')
    print('Solutions are PRACTICALLY INFEASIBLE if:')
    print('  1. A joint is near or at its hardware limit (risk of stall/damage).')
    print('  2. The configuration passes through a kinematic singularity.')
    print('  3. The arm collides with itself or the table.')
    print('  4. The required configuration cannot be reached from a nearby pose')
    print('     without passing through a joint limit or singularity.')
    print('Check the "Physical robot" column in the console output above.')


if __name__ == '__main__':
    main()

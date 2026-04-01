# Task 2.3 — Circle Trajectory

Commands the LeRobot 5-DoF arm to trace a horizontal circle using Jacobian-based inverse kinematics.

## Files

| File | Description |
|---|---|
| `robot_kinematics.py` | Forward kinematics, geometric Jacobian, and IK library |
| `simulate_circle.py` | Standalone simulation + plot (no ROS2 required) |
| `ik_circle.py` | ROS2 node for running on the real robot or simulator |
| `circle_simulation.png` | Output figure from the simulation |

## Circle parameters

| Parameter | Value |
|---|---|
| Centre | FK(q=0) = [0, 0.374, 0.237] m |
| Plane | Horizontal XY-plane (z = 0.237 m, constant) |
| Radius | 0.05 m |
| Period | 8 s per revolution |
| Revolutions | 2 (simulation) / 3 (ROS2 node) |

## Method

At each timestep the desired end-effector position is computed from the parametric circle:

```
x(t) = cx + R·cos(2π t / T)
y(t) = cy + R·sin(2π t / T)
z(t) = cz   (constant)
```

The joint angles are found using **position-only damped least squares (DLS) IK**,
using only the top 3 rows of the 6×5 geometric Jacobian:

```
dq = α · J_pos^T (J_pos · J_pos^T + λ²I)^{-1} · e_pos
```

The IK is **warm-started** from the previous timestep's joint solution to ensure
smooth, continuous motion. Tracking error: **mean = 0.07 mm, max = 0.07 mm**.

## Run the standalone simulation (no ROS2)

```bash
cd task_2_3
python3 simulate_circle.py
```

Requires: `numpy`, `matplotlib`

## Run on the robot

**Terminal 1** — start the hardware driver or simulator:
```bash
cd ~/edubot/ros_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Simulation:
ros2 launch lerobot sim_position.launch.py

# Real robot:
ros2 launch lerobot hw_position.launch.py
```

**Terminal 2** — start the circle trajectory:
```bash
source /opt/ros/jazzy/setup.bash
source ~/edubot/ros_ws/install/setup.bash

ros2 run python_controllers ik_circle
```

The arm will:
1. Move to `q = [0, 0, 0, 0, 0]`
2. Move to the circle start point
3. Trace the circle for 3 revolutions
4. Return to home

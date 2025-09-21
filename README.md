# SUSAG Project

> Last known-stable: **running as of *21/09/25*** (still needs tuning for smoother robot behavior).
> Simulator: **Gazebo Classic** (deprecated upstream, but used here intentionally).
> ROS distro: **ROS 2 Humble** (Ubuntu 22.04 recommended).


## 1) Requirements

* Ubuntu **22.04** + ROS 2 **Humble** (`/opt/ros/humble` available)
* **Gazebo Classic** (gazebo11)
* `colcon`, `vcstool`, `rosdep`

> Tip: avoid mixing Conda with ROS—PYTHONPATH conflicts are common.




## 2) Quick Start (fresh workspace)

```bash
# 0) ROS environment
source /opt/ros/humble/setup.bash

# 1) Create a workspace
mkdir -p ~/susag_ws/src
cd ~/susag_ws/src

# 2) Clone the repo
git clone <THIS_REPO_URL> 

# 3) Install system deps via rosdep
sudo apt update
rosdep update
rosdep install --from-paths src -r -i -y --rosdistro humble

# 4) Build
cd ~/susag_ws
colcon build --symlink-install

# 5) Source the overlay
source ~susag_ws/install/setup.bash
```

## 3) Running the Simulation

### Launch Gazebo Classic + robot

```bash
# New terminal
source /opt/ros/humble/setup.bash
source ~/susag_ws/install/setup.bash

ros2 launch susag_robot_description gazebo_world.launch.py
```

### Launch RViz & Nav stack

```bash
ros2 launch susag_nav2 slam.launch.py
```

### Common sanity checks

```bash
# See topics
ros2 topic list

# Check TF tree
ros2 run tf2_tools view_frames
# -> open frames.pdf and confirm map → odom → base_link → sensors are present

# Echo odometry / laser / depth topics
ros2 topic echo /odom
ros2 topic echo /scan
ros2 topic echo /susag/camera/depth/points
```

## 4) Branching & PR Checklist

1. **Branch** from `main`: `git checkout -b feat/<topic>`
2. Keep commits focused & descriptive.
3. **Build locally**: `colcon build` (no errors/warnings ideally)
4. **Run smoke tests**: launch sim, check `/tf`, `/odom`, sensors
5. **Document** changes in `docs/` and/or `README.md`
6. Open PR → request review → merge after two verifications

---

## 5) Known Status & Tuning Notes

* The robot **runs** but needs **further tuning** (e.g., Nav2 planner gains, AMCL params, controller limits).
* Using **Gazebo Classic** by design; migration to **Ignition/Gazebo Fortress+** is a future task.
* If you see TF issues like `base_link → odom` missing, verify your state estimator publishes `/odom` and that the TF tree is continuous.

---





#!/usr/bin/env python3
# Copyright 2026 SLIP project
#
# Amoeba-vs-vanilla MPPI benchmark harness for nav2 (ROS 2 Humble, Gazebo Classic).
#
# It does NOT swap controllers itself. You run nav2 with ONE params file
# (navigation_sim.yaml = vanilla, navigation_amoeba.yaml = smart amoeba),
# tag the run with --controller vanilla|amoeba, and this script drives the
# same start/goal/obstacle matrix and appends rows to a shared CSV. Run it
# once per controller; the CSV then holds both for a paired comparison.
#
# Metrics per trial (all derived from topics, no log scraping):
#   success        1 if NavigateToPose SUCCEEDED within timeout, else 0
#   time_s         wall time from goal accept to result (successful runs only meaningful)
#   path_len_m     integrated /odom travel
#   min_clear_m    min /scan range over the run (safety proxy)
#   stall_s        cumulative time |v|<STALL_V while still far from goal (mode-averaging proxy)
#   result_code    action result status (4=SUCCEEDED, 5=CANCELED, 6=ABORTED, 0=timeout)
#
# Usage:
#   # terminal A: bring up sim + nav2 (vanilla)
#   ros2 launch susag_nav2 navigation.launch.py sim:=true \
#        nav2_params:=<...>/navigation_sim.yaml
#   # (set the 2D Pose Estimate once so AMCL is localized)
#   # terminal B:
#   python3 benchmark_amoeba.py --controller vanilla --reps 10
#
#   # then relaunch nav2 with navigation_amoeba.yaml and:
#   python3 benchmark_amoeba.py --controller amoeba --reps 10
#
#   # compare:
#   python3 benchmark_amoeba.py --report        # prints summary table from the CSV

import argparse
import csv
import math
import os
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

# ---------------------------------------------------------------------------
# TUNE THESE to a clear corridor in YOUR map (maze_map_new: x[-8.51,9.99] y[-9.85,8.15]).
# START/GOAL should have a straight, obstacle-free global path between them so that
# the ONLY thing standing between the robot and the goal is the obstacle we spawn.
START = (-2.5, 1.0, 0.0)     # x, y, yaw(rad)  -- robot's maze_world spawn point
GOAL = (2.3, -1.3, 0.0)      # x, y, yaw(rad)  -- PLACEHOLDER: set to a clear corridor end
ROBOT_ENTITY = "susag_updated_model"
GOAL_TOL = 0.35             # m, matches general_goal_checker xy_goal_tolerance-ish
TIMEOUT_S = 60.0            # per-trial ceiling; exceeding it = failure (code 0)
STALL_V = 0.03             # m/s below which we count time as "stalled"
STALL_NEAR_GOAL = 0.6      # don't count stall within this dist of goal (it's just settling)

# Scenario obstacles as (dx, dy, size) offsets from a reference point on the
# ACTUAL planned path, in the path frame (dx=along the path tangent, dy=left).
# At runtime the harness plans START->GOAL, takes the midpoint of that path and
# its tangent, and maps these offsets to world coords -- so the box lands ON the
# route even though NavFn curves around maze walls. Empty list = open course.
SCENARIOS = {
    # S0: nothing on the path -> amoeba should ~= vanilla (must not regress badly)
    "S0_open": [],
    # S1: single pillar dead-centre on the path -> the classic mode-averaging trap
    "S1_pillar": [(0.0, 0.0, 0.4)],
    # S2: symmetric fork -> two boxes straddling the path, gap in the middle
    "S2_fork": [(0.0, 0.7, 0.5), (0.0, -0.7, 0.5)],
    # S3: narrow gap -> tighter straddle, forces "don't average into the wall"
    "S3_gap": [(0.0, 0.55, 0.5), (0.0, -0.55, 0.5)],
    # S4: dead-end pocket -> back wall + two side walls, tests walled-reverse
    "S4_pocket": [(0.6, 0.0, 0.6), (0.0, 0.75, 0.6), (0.0, -0.75, 0.6)],
}

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
CSV_FIELDS = ["controller", "scenario", "rep", "success", "time_s",
              "path_len_m", "min_clear_m", "stall_s", "result_code"]


def box_sdf(size):
    h = size / 2.0
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="bench_box">
    <static>true</static>
    <link name="link">
      <collision name="c">
        <geometry><box><size>{size} {size} 0.6</size></box></geometry>
      </collision>
      <visual name="v">
        <geometry><box><size>{size} {size} 0.6</size></box></geometry>
        <material><ambient>0.8 0.2 0.2 1</ambient><diffuse>0.8 0.2 0.2 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class Bench(Node):
    def __init__(self, controller, reps, scenarios):
        super().__init__("amoeba_bench")
        self.controller = controller
        self.reps = reps
        self.scenarios = scenarios

        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, "/odom", self._odom_cb, best_effort)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, best_effort)

        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.planner = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete = self.create_client(DeleteEntity, "/delete_entity")

        # live state
        self.pose = None            # (x, y, yaw)
        self.vlin = 0.0
        self.min_clear = float("inf")
        self.path_len = 0.0
        self.stall_s = 0.0
        self._last_xy = None
        self._last_t = None

    # -- callbacks ----------------------------------------------------------
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        v = msg.twist.twist.linear
        self.vlin = math.hypot(v.x, v.y)

    def _scan_cb(self, msg):
        for r in msg.ranges:
            if msg.range_min < r < self.min_clear:
                self.min_clear = r

    # -- helpers ------------------------------------------------------------
    def _reset_run_metrics(self):
        self.min_clear = float("inf")
        self.path_len = 0.0
        self.stall_s = 0.0
        self._last_xy = None
        self._last_t = None

    def _accumulate(self, goal_xy):
        """Call each spin tick during a run to integrate path/stall."""
        if self.pose is None:
            return
        now = time.time()
        x, y, _ = self.pose
        if self._last_xy is not None:
            self.path_len += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            dt = now - self._last_t
            d_goal = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            if self.vlin < STALL_V and d_goal > STALL_NEAR_GOAL:
                self.stall_s += dt
        self._last_xy = (x, y)
        self._last_t = now

    def wait_services(self):
        self.get_logger().info("waiting for nav2 + gazebo services...")
        self.nav.wait_for_server()
        self.spawn.wait_for_service()
        self.delete.wait_for_service()
        # wait for first odom
        while rclpy.ok() and self.pose is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("services up, odom flowing.")

    def compute_path(self):
        """Plan START->GOAL and return the path as a list of (x, y). [] on failure."""
        goal = ComputePathToPose.Goal()
        goal.use_start = True
        for tgt, dst in ((START, goal.start), (GOAL, goal.goal)):
            dst.header.frame_id = "map"
            dst.header.stamp = self.get_clock().now().to_msg()
            dst.pose.position.x = float(tgt[0])
            dst.pose.position.y = float(tgt[1])
            dst.pose.orientation.z = math.sin(tgt[2] / 2.0)
            dst.pose.orientation.w = math.cos(tgt[2] / 2.0)
        if not self.planner.wait_for_server(timeout_sec=5.0):
            return []
        send_fut = self.planner.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        handle = send_fut.result()
        if handle is None or not handle.accepted:
            return []
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=10.0)
        res = res_fut.result()
        if res is None:
            return []
        return [(p.pose.position.x, p.pose.position.y) for p in res.result.path.poses]

    def resolve_offsets(self, offsets):
        """Map (dx, dy, size) path-frame offsets to world (wx, wy, size) triples,
        anchored at the midpoint of the planned path with the path tangent as +x.
        Falls back to the straight START->GOAL midpoint if planning fails."""
        if not offsets:
            return []
        path = self.compute_path()
        if len(path) >= 2:
            m = len(path) // 2
            px, py = path[m]
            nx, ny = path[min(m + 1, len(path) - 1)]
            th = math.atan2(ny - py, nx - px)
        else:
            px = (START[0] + GOAL[0]) / 2.0
            py = (START[1] + GOAL[1]) / 2.0
            th = math.atan2(GOAL[1] - START[1], GOAL[0] - START[0])
            self.get_logger().warn("path planning failed; placing on straight-line midpoint")
        c, s = math.cos(th), math.sin(th)
        out = []
        for dx, dy, size in offsets:
            wx = px + dx * c - dy * s
            wy = py + dx * s + dy * c
            out.append((wx, wy, size))
        return out

    def spawn_obstacles(self, placed):
        """placed: list of (wx, wy, size) world triples from resolve_offsets()."""
        names = []
        for i, (wx, wy, size) in enumerate(placed):
            req = SpawnEntity.Request()
            req.name = f"bench_box_{i}"
            req.xml = box_sdf(size)
            req.initial_pose.position.x = float(wx)
            req.initial_pose.position.y = float(wy)
            req.initial_pose.position.z = 0.3
            fut = self.spawn.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            names.append(req.name)
        time.sleep(0.5)  # let costmap register them
        return names

    def delete_obstacles(self, names):
        for n in names:
            req = DeleteEntity.Request()
            req.name = n
            fut = self.delete.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        time.sleep(0.5)

    def send_goal(self, xy_yaw, timeout_s, record=True):
        """Drive to xy_yaw. Returns (success, elapsed, result_code). Records metrics if record."""
        gx, gy, gyaw = xy_yaw
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(gx)
        ps.pose.position.y = float(gy)
        ps.pose.orientation.z = math.sin(gyaw / 2.0)
        ps.pose.orientation.w = math.cos(gyaw / 2.0)
        goal.pose = ps

        send_fut = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        handle = send_fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("goal rejected")
            return False, 0.0, -1

        if record:
            self._reset_run_metrics()
        result_fut = handle.get_result_async()
        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if record:
                self._accumulate((gx, gy))
            if result_fut.done():
                status = result_fut.result().status
                return status == 4, time.time() - t0, status
            if time.time() - t0 > timeout_s:
                handle.cancel_goal_async()
                rclpy.spin_once(self, timeout_sec=0.5)
                return False, time.time() - t0, 0
        return False, time.time() - t0, -1

    def at_start(self):
        if self.pose is None:
            return False
        return math.hypot(self.pose[0] - START[0], self.pose[1] - START[1]) < GOAL_TOL + 0.2

    def go_home(self):
        if self.at_start():
            return
        self.get_logger().info("returning to start...")
        self.send_goal(START, timeout_s=TIMEOUT_S, record=False)

    # -- main matrix --------------------------------------------------------
    def run(self):
        self.wait_services()
        rows = []
        for scen in self.scenarios:
            boxes = SCENARIOS[scen]
            for rep in range(self.reps):
                self.get_logger().info(f"=== {self.controller} | {scen} | rep {rep} ===")
                self.go_home()
                # Resolve obstacle positions on the ACTUAL planned path (post go_home,
                # so the plan starts from the robot at START), then spawn.
                placed = self.resolve_offsets(boxes)
                names = self.spawn_obstacles(placed)
                ok, t, code = self.send_goal(GOAL, TIMEOUT_S)
                row = {
                    "controller": self.controller, "scenario": scen, "rep": rep,
                    "success": int(ok), "time_s": round(t, 2),
                    "path_len_m": round(self.path_len, 2),
                    "min_clear_m": round(self.min_clear, 3) if self.min_clear != float("inf") else -1,
                    "stall_s": round(self.stall_s, 2), "result_code": code,
                }
                rows.append(row)
                self.get_logger().info(
                    f"  -> success={ok} t={t:.1f}s path={self.path_len:.1f}m "
                    f"clear={row['min_clear_m']}m stall={self.stall_s:.1f}s")
                self.delete_obstacles(names)
                self._append_csv(row)
        self.go_home()
        self.get_logger().info(f"done. {len(rows)} trials -> {CSV_PATH}")

    def _append_csv(self, row):
        new = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def print_report():
    if not os.path.exists(CSV_PATH):
        print(f"no results at {CSV_PATH}")
        return
    import statistics as st
    rows = list(csv.DictReader(open(CSV_PATH)))
    # group by (controller, scenario)
    groups = {}
    for r in rows:
        groups.setdefault((r["controller"], r["scenario"]), []).append(r)

    def agg(rs):
        n = len(rs)
        succ = [r for r in rs if r["success"] == "1"]
        sr = len(succ) / n if n else 0
        t = [float(r["time_s"]) for r in succ]
        stall = [float(r["stall_s"]) for r in rs]
        clear = [float(r["min_clear_m"]) for r in rs if float(r["min_clear_m"]) >= 0]
        pl = [float(r["path_len_m"]) for r in succ]
        return (n, sr,
                (st.mean(t) if t else float("nan")),
                (st.mean(pl) if pl else float("nan")),
                (st.mean(clear) if clear else float("nan")),
                (st.mean(stall) if stall else float("nan")))

    scenarios = sorted({k[1] for k in groups})
    controllers = sorted({k[0] for k in groups})
    hdr = f"{'scenario':<12}{'ctrl':<10}{'n':>3}{'succ%':>7}{'t_s':>8}{'path_m':>8}{'clear_m':>9}{'stall_s':>9}"
    print(hdr)
    print("-" * len(hdr))
    for scen in scenarios:
        for ctrl in controllers:
            if (ctrl, scen) not in groups:
                continue
            n, sr, t, pl, clr, stall = agg(groups[(ctrl, scen)])
            print(f"{scen:<12}{ctrl:<10}{n:>3}{sr*100:>6.0f}%"
                  f"{t:>8.1f}{pl:>8.1f}{clr:>9.2f}{stall:>9.1f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["vanilla", "amoeba"], help="tag for this run")
    ap.add_argument("--reps", type=int, default=10, help="trials per scenario")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                    help=f"subset of {list(SCENARIOS.keys())}")
    ap.add_argument("--report", action="store_true", help="just print the summary table and exit")
    args = ap.parse_args()

    if args.report:
        print_report()
        return
    if not args.controller:
        ap.error("--controller is required (vanilla|amoeba) unless --report")

    rclpy.init()
    node = Bench(args.controller, args.reps, args.scenarios)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

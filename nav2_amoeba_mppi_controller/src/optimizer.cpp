// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_amoeba_mppi_controller/optimizer.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace amoeba
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();

  // Debug-viz publisher. Activated here (not via the wrapper's on_activate) so
  // it is ready before the first control cycle; it only emits while the
  // controller is active, since computeAmoebaModes() runs only from evalControl.
  amoeba_debug_pub_ =
    node->create_publisher<visualization_msgs::msg::MarkerArray>("/amoeba_debug", 1);
  amoeba_debug_pub_->on_activate();

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

  reset();
}

void Optimizer::shutdown()
{
  noise_generator_.shutdown();
}

void Optimizer::getParams()
{
  std::string motion_model_name;

  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);
  getParam(s.batch_size, "batch_size", 1000);
  getParam(s.iteration_count, "iteration_count", 1);
  getParam(s.temperature, "temperature", 0.3f);
  getParam(s.gamma, "gamma", 0.015f);
  getParam(s.base_constraints.vx_max, "vx_max", 0.5);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35);
  getParam(s.base_constraints.vy, "vy_max", 0.5);
  getParam(s.base_constraints.wz, "wz_max", 1.9);
  getParam(s.sampling_std.vx, "vx_std", 0.2);
  getParam(s.sampling_std.vy, "vy_std", 0.2);
  getParam(s.sampling_std.wz, "wz_std", 0.4);
  getParam(s.retry_attempt_limit, "retry_attempt_limit", 1);

  getParam(s.amoeba_bias_strength, "amoeba_bias_strength", 0.6f);
  getParam(s.amoeba_bias_gain, "amoeba_bias_gain", 1.5f);
  getParam(s.amoeba_lookahead_dist, "amoeba_lookahead_dist", 0.6f);
  getParam(s.amoeba_scan_range, "amoeba_scan_range", 2.5f);
  getParam(s.amoeba_arc, "amoeba_arc", 1.57f);
  getParam(s.amoeba_num_rays, "amoeba_num_rays", 37);
  getParam(s.amoeba_goal_dist, "amoeba_goal_dist", 1.0f);
  getParam(s.amoeba_debug, "amoeba_debug", false);
  RCLCPP_INFO(
    logger_,
    "[AMOEBA] shape-conditioned sampling (scan=%.2fm arc=%.2frad rays=%d bias=%.2f)",
    s.amoeba_scan_range, s.amoeba_arc, s.amoeba_num_rays, s.amoeba_bias_strength);

  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  parameters_handler_->addPostCallback([this]() {reset();});

  double controller_frequency;
  getParentParam(controller_frequency, "controller_frequency", 0.0, ParameterType::Static);
  setOffset(controller_frequency);
}

void Optimizer::setOffset(double controller_frequency)
{
  const double controller_period = 1.0 / controller_frequency;
  constexpr double eps = 1e-6;

  if ((controller_period + eps) < settings_.model_dt) {
    RCLCPP_WARN(
      logger_,
      "Controller period is less then model dt, consider setting it equal");
  } else if (abs(controller_period - settings_.model_dt) < eps) {
    RCLCPP_INFO(
      logger_,
      "Controller period is equal to model dt. Control sequence "
      "shifting is ON");
    settings_.shift_control_sequence = true;
  } else {
    throw std::runtime_error(
            "Controller period more then model dt, set it equal to model dt");
  }
}

void Optimizer::reset()
{
  state_.reset(settings_.batch_size, settings_.time_steps);
  control_sequence_.reset(settings_.time_steps);
  control_history_[0] = {0.0, 0.0, 0.0};
  control_history_[1] = {0.0, 0.0, 0.0};
  control_history_[2] = {0.0, 0.0, 0.0};
  control_history_[3] = {0.0, 0.0, 0.0};

  settings_.constraints = settings_.base_constraints;

  costs_ = xt::zeros<float>({settings_.batch_size});
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());
  RCLCPP_INFO(logger_, "Optimizer reset");
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  prepare(robot_pose, robot_speed, plan, goal_checker);

  do {
    optimize();
  } while (fallback(critics_data_.fail_flag));

  utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);
  auto control = getControlFromSequenceAsTwist(plan.header.stamp);

  if (settings_.shift_control_sequence) {
    shiftControlSequence();
  }

  return control;
}

void Optimizer::optimize()
{
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    generateNoisedTrajectories();
    critic_manager_.evalTrajectoriesScores(critics_data_);
    updateControlSequence();
  }
}

bool Optimizer::fallback(bool fail)
{
  static size_t counter = 0;

  if (!fail) {
    counter = 0;
    return false;
  }

  reset();

  if (++counter > settings_.retry_attempt_limit) {
    counter = 0;
    throw std::runtime_error("Optimizer fail to compute path");
  }

  return true;
}

void Optimizer::prepare(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  state_.pose = robot_pose;
  state_.speed = robot_speed;
  path_ = utils::toTensor(plan);
  costs_.fill(0);

  critics_data_.fail_flag = false;
  critics_data_.goal_checker = goal_checker;
  critics_data_.motion_model = motion_model_;
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
}

void Optimizer::shiftControlSequence()
{
  using namespace xt::placeholders;  // NOLINT
  control_sequence_.vx = xt::roll(control_sequence_.vx, -1);
  control_sequence_.wz = xt::roll(control_sequence_.wz, -1);


  xt::view(control_sequence_.vx, -1) =
    xt::view(control_sequence_.vx, -2);

  xt::view(control_sequence_.wz, -1) =
    xt::view(control_sequence_.wz, -2);


  if (isHolonomic()) {
    control_sequence_.vy = xt::roll(control_sequence_.vy, -1);
    xt::view(control_sequence_.vy, -1) =
      xt::view(control_sequence_.vy, -2);
  }
}

void Optimizer::generateNoisedTrajectories()
{
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  applyAmoebaBias();  // amoeba: seed shape-derived wrap modes before rollout
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
}

float Optimizer::amoebaCostAt(float wx, float wy) const
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(wx, wy, mx, my)) {
    return 0.0f;  // off-map -> treat as free
  }
  const unsigned char c = costmap_->getCost(mx, my);
  if (c == nav2_costmap_2d::NO_INFORMATION) {
    return 0.0f;  // unknown -> treat as free
  }
  return static_cast<float>(c);
}

float Optimizer::buildWrapSequence(
  float rx, float ry, float ryaw, float wx1, float wy1, float wgx, float wgy, float prox,
  std::vector<float> & vx, std::vector<float> & wz,
  std::vector<float> & px, std::vector<float> & py)
{
  const auto & s = settings_;
  const unsigned int T = std::max(1u, s.time_steps);
  vx.assign(T, 0.0f);
  wz.assign(T, 0.0f);
  px.assign(T, rx);
  py.assign(T, ry);

  const float dt = s.model_dt;
  const float kLethal = static_cast<float>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  constexpr float kReach = 0.3f;   // aim flips from skirt point to goal within this dist
  constexpr float kHit = 1.0e6f;   // a clipped lethal cell -> invalid mode

  float x = rx, y = ry, yaw = ryaw;
  bool reached_skirt = false;
  float cost_sum = 0.0f;
  for (unsigned int t = 0; t < T; ++t) {
    // Aim at the corner-skirt point until reached, then re-aim at the goal:
    // this is what curves the path AROUND the obstacle instead of a fixed arc.
    if (!reached_skirt && std::hypot(wx1 - x, wy1 - y) <= kReach) {
      reached_skirt = true;
    }
    const float ax = reached_skirt ? wgx : wx1;
    const float ay = reached_skirt ? wgy : wy1;
    const float aim = std::atan2(ay - y, ax - x);
    const float he = static_cast<float>(angles::shortest_angular_distance(yaw, aim));

    const float w = std::clamp(s.amoeba_bias_gain * he, -s.constraints.wz, s.constraints.wz);
    const float v = std::clamp(
      s.constraints.vx_max * std::max(0.0f, std::cos(he)) * prox, 0.0f, s.constraints.vx_max);
    vx[t] = v;
    wz[t] = w;

    // Integrate the diff-drive model one step.
    yaw += w * dt;
    x += v * std::cos(yaw) * dt;
    y += v * std::sin(yaw) * dt;
    px[t] = x;
    py[t] = y;

    const float c = amoebaCostAt(x, y);
    cost_sum += (c >= kLethal) ? kHit : c;
  }
  return cost_sum;
}

void Optimizer::computeAmoebaModes()
{
  const auto & s = settings_;
  amoeba_active_ = false;
  amoeba_two_modes_ = false;
  amoeba_reverse_ = false;
  amoeba_alloc_l_ = 1.0f;
  amoeba_bearing_l_ = amoeba_bearing_r_ = 0.0f;
  amoeba_seq_vx_l_.clear(); amoeba_seq_wz_l_.clear();
  amoeba_seq_vx_r_.clear(); amoeba_seq_wz_r_.clear();
  amoeba_path_lx_.clear(); amoeba_path_ly_.clear();
  amoeba_path_rx_.clear(); amoeba_path_ry_.clear();
  if (s.amoeba_debug) {
    dbg_ray_range_.clear();
    dbg_ray_bearing_.clear();
  }
  if (costmap_ == nullptr) {
    return;
  }

  const float rx = static_cast<float>(state_.pose.pose.position.x);
  const float ry = static_cast<float>(state_.pose.pose.position.y);
  const float ryaw = tf2::getYaw(state_.pose.pose.orientation);
  dbg_rx_ = rx;
  dbg_ry_ = ry;
  dbg_ryaw_ = ryaw;

  const float kLethal = static_cast<float>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);

  // "Toward-goal" point + bearing: first path point >= lookahead away, else last.
  const std::size_t path_size = path_.x.shape(0);

  // Endgame guard: within amoeba_goal_dist of the goal (last path point),
  // disable the amoeba and defer to plain MPPI + GoalCritic. Otherwise the
  // gap-scan keeps injecting wrap/reverse modes at the goal (nearby walls read
  // as blocked, reverse fights the goal pull) and the robot oscillates.
  if (path_size >= 1) {
    const float ex = path_.x(path_size - 1) - rx;
    const float ey = path_.y(path_size - 1) - ry;
    if (ex * ex + ey * ey < s.amoeba_goal_dist * s.amoeba_goal_dist) {
      return;  // amoeba_active_ already false, all mode buffers already cleared
    }
  }

  float goal_bearing = ryaw;
  float gx = rx + std::cos(ryaw);   // fallback: a metre straight ahead
  float gy = ry + std::sin(ryaw);
  if (path_size >= 2) {
    std::size_t idx = path_size - 1;
    const float ld2 = s.amoeba_lookahead_dist * s.amoeba_lookahead_dist;
    for (std::size_t i = 0; i < path_size; ++i) {
      const float dx = path_.x(i) - rx;
      const float dy = path_.y(i) - ry;
      if (dx * dx + dy * dy >= ld2) {
        idx = i;
        break;
      }
    }
    gx = path_.x(idx);
    gy = path_.y(idx);
    goal_bearing = std::atan2(gy - ry, gx - rx);
  }

  // Polar gap-scan: cast a fan of rays over the forward arc and record the
  // range to the first lethal cell. This IS the obstacle silhouette.
  const int num_rays = std::max(3, s.amoeba_num_rays);
  const float arc = s.amoeba_arc;
  const float scan = s.amoeba_scan_range;
  const float step = std::max(static_cast<float>(costmap_->getResolution()), 0.05f);
  const float dtheta = (2.0f * arc) / static_cast<float>(num_rays - 1);

  std::vector<float> ray_range(num_rays, scan + 1.0f);  // > scan means "free ray"
  std::vector<float> ray_bearing(num_rays);
  for (int i = 0; i < num_rays; ++i) {
    const float b = ryaw - arc + dtheta * static_cast<float>(i);
    ray_bearing[i] = b;
    const float cb = std::cos(b);
    const float sb = std::sin(b);
    for (float d = step; d <= scan; d += step) {
      if (amoebaCostAt(rx + d * cb, ry + d * sb) >= kLethal) {
        ray_range[i] = d;
        break;
      }
    }
  }
  auto blocked = [&](int i) {return ray_range[i] <= scan;};

  if (s.amoeba_debug) {
    dbg_ray_range_ = ray_range;
    dbg_ray_bearing_ = ray_bearing;
  }

  // Ray closest to the goal bearing = the direction we actually want to go.
  int gi = 0;
  float best = 1e9f;
  for (int i = 0; i < num_rays; ++i) {
    const float e = std::fabs(
      static_cast<float>(angles::shortest_angular_distance(ray_bearing[i], goal_bearing)));
    if (e < best) {
      best = e;
      gi = i;
    }
  }

  // Clear path -> plain MPPI; reset the hysteresis commitment.
  if (!blocked(gi)) {
    amoeba_last_side_ = -1;
    return;
  }
  amoeba_active_ = true;

  // Grow the blocked sector outward from gi to find its two edges.
  int hi = gi;
  int lo = gi;
  while (hi + 1 < num_rays && blocked(hi + 1)) {++hi;}
  while (lo - 1 >= 0 && blocked(lo - 1)) {--lo;}
  const bool gap_hi = (hi + 1 < num_rays);  // a free ray just past the high-index edge
  const bool gap_lo = (lo - 1 >= 0);        // a free ray just past the low-index edge

  // Nearest obstacle range in the sector -> slow down the closer it is.
  float nearest = scan;
  for (int i = lo; i <= hi; ++i) {
    nearest = std::min(nearest, ray_range[i]);
  }
  const float prox = std::clamp(nearest / scan, 0.2f, 1.0f);
  const unsigned int T = std::max(1u, s.time_steps);

  auto set_reverse = [&]() {
      amoeba_seq_vx_l_.assign(T, s.constraints.vx_min);
      amoeba_seq_wz_l_.assign(T, 0.0f);
      amoeba_seq_vx_r_.clear();
      amoeba_seq_wz_r_.clear();
      amoeba_path_lx_.clear();
      amoeba_path_ly_.clear();
      amoeba_path_rx_.clear();
      amoeba_path_ry_.clear();
      amoeba_reverse_ = true;
      amoeba_two_modes_ = false;
      amoeba_alloc_l_ = 1.0f;
      amoeba_last_side_ = -1;
      amoeba_bearing_l_ = ryaw + static_cast<float>(M_PI);  // arrow points backward
    };

  // Entire scanned arc blocked: no local way around (wall / concave trap).
  if (!gap_hi && !gap_lo) {
    set_reverse();
    return;
  }

  // Build a time-varying wrap path for a free-edge bearing; returns its cost.
  const float skirt_dist = std::min(scan, nearest + 0.6f);
  auto make_mode = [&](float edge_bearing,
      std::vector<float> & sv, std::vector<float> & sw,
      std::vector<float> & pxs, std::vector<float> & pys) -> float {
      const float wx1 = rx + skirt_dist * std::cos(edge_bearing);
      const float wy1 = ry + skirt_dist * std::sin(edge_bearing);
      return buildWrapSequence(rx, ry, ryaw, wx1, wy1, gx, gy, prox, sv, sw, pxs, pys);
    };

  const float bearing_hi = ray_bearing[std::min(hi + 1, num_rays - 1)];
  const float bearing_lo = ray_bearing[std::max(lo - 1, 0)];
  constexpr float kHit = 1.0e6f;

  if (gap_hi && gap_lo) {
    // Two candidate tangents -> build BOTH, score by costmap cost, allocate the
    // batch to the cheaper side (this is the cost-aware, non-P-controller part).
    const float score_l = make_mode(
      bearing_hi, amoeba_seq_vx_l_, amoeba_seq_wz_l_, amoeba_path_lx_, amoeba_path_ly_);
    const float score_r = make_mode(
      bearing_lo, amoeba_seq_vx_r_, amoeba_seq_wz_r_, amoeba_path_rx_, amoeba_path_ry_);
    amoeba_bearing_l_ = bearing_hi;
    amoeba_bearing_r_ = bearing_lo;
    const bool valid_l = score_l < kHit;
    const bool valid_r = score_r < kHit;

    if (valid_l && valid_r) {
      // Lower-cost side gets the larger share of the biased batch.
      const float denom = score_l + score_r;
      amoeba_alloc_l_ = (denom > 1e-3f) ? std::clamp(score_r / denom, 0.1f, 0.9f) : 0.5f;
      // Hysteresis: if the sides are close, keep last cycle's committed side so
      // the robot doesn't dither left/right frame to frame.
      const float rel = std::fabs(score_l - score_r) / std::max(1.0f, denom);
      if (rel < 0.15f && amoeba_last_side_ >= 0) {
        amoeba_alloc_l_ = (amoeba_last_side_ == 0) ?
          std::max(amoeba_alloc_l_, 0.6f) : std::min(amoeba_alloc_l_, 0.4f);
      }
      amoeba_last_side_ = (amoeba_alloc_l_ >= 0.5f) ? 0 : 1;
      amoeba_two_modes_ = true;
    } else if (valid_l) {
      amoeba_seq_vx_r_.clear();
      amoeba_seq_wz_r_.clear();
      amoeba_path_rx_.clear();
      amoeba_path_ry_.clear();
      amoeba_alloc_l_ = 1.0f;
      amoeba_last_side_ = 0;
    } else if (valid_r) {
      // Move the surviving right mode into the left slot (single-mode seeding).
      amoeba_seq_vx_l_.swap(amoeba_seq_vx_r_);
      amoeba_seq_wz_l_.swap(amoeba_seq_wz_r_);
      amoeba_path_lx_.swap(amoeba_path_rx_);
      amoeba_path_ly_.swap(amoeba_path_ry_);
      amoeba_seq_vx_r_.clear();
      amoeba_seq_wz_r_.clear();
      amoeba_path_rx_.clear();
      amoeba_path_ry_.clear();
      amoeba_bearing_l_ = bearing_lo;
      amoeba_alloc_l_ = 1.0f;
      amoeba_last_side_ = 1;
    } else {
      set_reverse();  // both tangents clip -> no viable wrap
    }
  } else {
    // Only one side clears -> single wrap mode toward it; reverse if it clips.
    const float bearing = gap_hi ? bearing_hi : bearing_lo;
    const float score = make_mode(
      bearing, amoeba_seq_vx_l_, amoeba_seq_wz_l_, amoeba_path_lx_, amoeba_path_ly_);
    amoeba_bearing_l_ = bearing;
    amoeba_alloc_l_ = 1.0f;
    amoeba_last_side_ = -1;
    if (score >= kHit) {
      set_reverse();
    }
  }
}

void Optimizer::applyAmoebaBias()
{
  computeAmoebaModes();
  if (settings_.amoeba_debug) {
    publishAmoebaDebug();  // draw the scan + decision every cycle, active or not
  }
  if (!amoeba_active_) {
    return;  // clear path -> plain MPPI sampling
  }

  const auto & s = settings_;
  const float frac = std::clamp(s.amoeba_bias_strength, 0.0f, 1.0f);
  const unsigned int n_biased =
    std::min(s.batch_size, static_cast<unsigned int>(frac * static_cast<float>(s.batch_size)));
  if (n_biased == 0 || amoeba_seq_vx_l_.empty()) {
    return;
  }

  // Cost-aware split: alloc_l of the biased rows follow the left/single mode,
  // the rest follow the right mode. Each row is re-centered onto its mode's
  // TIME-VARYING control, REUSING that row's noise:
  //   cvx(row,t) += u_mode(t) - nominal(t)  =>  cvx(row,t) = u_mode(t) + eps
  // The two wrap paths stay distinct sample populations (never mode-averaged).
  const unsigned int n_left = amoeba_two_modes_ ?
    static_cast<unsigned int>(std::clamp(amoeba_alloc_l_, 0.0f, 1.0f) *
    static_cast<float>(n_biased)) : n_biased;

  for (unsigned int row = 0; row < n_biased; ++row) {
    const bool left = (row < n_left);
    const auto & sv = left ? amoeba_seq_vx_l_ : amoeba_seq_vx_r_;
    const auto & sw = left ? amoeba_seq_wz_l_ : amoeba_seq_wz_r_;
    if (sv.empty()) {
      continue;
    }
    for (unsigned int t = 0; t < s.time_steps; ++t) {
      const std::size_t k = std::min<std::size_t>(t, sv.size() - 1);
      state_.cvx(row, t) += sv[k] - control_sequence_.vx(t);
      state_.cwz(row, t) += sw[k] - control_sequence_.wz(t);
    }
  }
}

void Optimizer::publishAmoebaDebug()
{
  if (!amoeba_debug_pub_ || amoeba_debug_pub_->get_subscription_count() == 0) {
    return;  // nobody listening -> skip the marker work
  }
  auto node = parent_.lock();
  if (!node) {
    return;
  }
  const auto stamp = node->now();
  const std::string frame = costmap_ros_->getGlobalFrameID();
  const float scan = settings_.amoeba_scan_range;
  constexpr double kZ = 0.05;

  visualization_msgs::msg::MarkerArray arr;

  // Clear last cycle's markers so stale rays/arrows don't linger.
  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = frame;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  auto mk_point = [](double x, double y, double z) {
      geometry_msgs::msg::Point p;
      p.x = x; p.y = y; p.z = z;
      return p;
    };
  auto mk_color = [](float r, float g, float b, float a = 1.0f) {
      std_msgs::msg::ColorRGBA c;
      c.r = r; c.g = g; c.b = b; c.a = a;
      return c;
    };

  // 1) Gap-scan rays: red where a lethal cell was hit, green where clear.
  if (!dbg_ray_range_.empty()) {
    visualization_msgs::msg::Marker rays;
    rays.header.frame_id = frame;
    rays.header.stamp = stamp;
    rays.ns = "amoeba_scan";
    rays.id = 0;
    rays.type = visualization_msgs::msg::Marker::LINE_LIST;
    rays.action = visualization_msgs::msg::Marker::ADD;
    rays.scale.x = 0.02;
    rays.pose.orientation.w = 1.0;
    for (std::size_t i = 0; i < dbg_ray_range_.size(); ++i) {
      const bool hit = dbg_ray_range_[i] <= scan;
      const float d = hit ? dbg_ray_range_[i] : scan;
      const float b = dbg_ray_bearing_[i];
      const auto col = hit ? mk_color(0.95f, 0.20f, 0.15f) : mk_color(0.20f, 0.85f, 0.35f);
      rays.points.push_back(mk_point(dbg_rx_, dbg_ry_, kZ));
      rays.points.push_back(mk_point(dbg_rx_ + d * std::cos(b), dbg_ry_ + d * std::sin(b), kZ));
      rays.colors.push_back(col);
      rays.colors.push_back(col);
    }
    arr.markers.push_back(rays);
  }

  // 2a) A short arrow for the reverse/walled case (no path to draw).
  auto add_arrow = [&](int id, float bearing, float r, float g, float b) {
      visualization_msgs::msg::Marker a;
      a.header.frame_id = frame;
      a.header.stamp = stamp;
      a.ns = "amoeba_mode";
      a.id = id;
      a.type = visualization_msgs::msg::Marker::ARROW;
      a.action = visualization_msgs::msg::Marker::ADD;
      a.scale.x = 0.06;
      a.scale.y = 0.14;
      a.scale.z = 0.18;
      a.color = mk_color(r, g, b);
      const float len = 1.0f;
      a.points.push_back(mk_point(dbg_rx_, dbg_ry_, kZ));
      a.points.push_back(
        mk_point(dbg_rx_ + len * std::cos(bearing), dbg_ry_ + len * std::sin(bearing), kZ));
      arr.markers.push_back(a);
    };

  // 2b) The actual TIME-VARYING wrap path the controller proposed, as a strip.
  auto add_path = [&](int id, const std::vector<float> & xs, const std::vector<float> & ys,
      float r, float g, float b) {
      if (xs.size() < 2) {return;}
      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame;
      m.header.stamp = stamp;
      m.ns = "amoeba_mode";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = 0.05;
      m.color = mk_color(r, g, b);
      m.pose.orientation.w = 1.0;
      m.points.push_back(mk_point(dbg_rx_, dbg_ry_, kZ));
      for (std::size_t i = 0; i < xs.size(); ++i) {
        m.points.push_back(mk_point(xs[i], ys[i], kZ));
      }
      arr.markers.push_back(m);
    };

  std::string state;
  if (!amoeba_active_) {
    state = "AMOEBA: clear";
  } else if (amoeba_reverse_) {
    add_arrow(1, amoeba_bearing_l_, 0.70f, 0.20f, 0.90f);  // purple = reverse
    state = "AMOEBA: reverse (walled)";
  } else if (amoeba_two_modes_) {
    add_path(1, amoeba_path_lx_, amoeba_path_ly_, 0.20f, 0.45f, 1.00f);  // blue = left wrap
    add_path(2, amoeba_path_rx_, amoeba_path_ry_, 1.00f, 0.55f, 0.10f);  // orange = right wrap
    char buf[48];
    std::snprintf(buf, sizeof(buf), "AMOEBA: wrap x2 (L=%.0f%%)", amoeba_alloc_l_ * 100.0f);
    state = buf;
  } else {
    add_path(1, amoeba_path_lx_, amoeba_path_ly_, 0.20f, 0.45f, 1.00f);  // single wrap
    state = "AMOEBA: wrap x1";
  }

  // 3) State label floating above the robot.
  visualization_msgs::msg::Marker txt;
  txt.header.frame_id = frame;
  txt.header.stamp = stamp;
  txt.ns = "amoeba_state";
  txt.id = 0;
  txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  txt.action = visualization_msgs::msg::Marker::ADD;
  txt.pose.position = mk_point(dbg_rx_, dbg_ry_, 0.7);
  txt.pose.orientation.w = 1.0;
  txt.scale.z = 0.25;
  txt.color = mk_color(1.0f, 1.0f, 1.0f);
  txt.text = state;
  arr.markers.push_back(txt);

  amoeba_debug_pub_->publish(arr);
}

bool Optimizer::isHolonomic() const {return motion_model_->isHolonomic();}

void Optimizer::applyControlSequenceConstraints()
{
  auto & s = settings_;

  if (isHolonomic()) {
    control_sequence_.vy = xt::clip(control_sequence_.vy, -s.constraints.vy, s.constraints.vy);
  }

  control_sequence_.vx = xt::clip(control_sequence_.vx, s.constraints.vx_min, s.constraints.vx_max);
  control_sequence_.wz = xt::clip(control_sequence_.wz, -s.constraints.wz, s.constraints.wz);

  motion_model_->applyConstraints(control_sequence_);
}

void Optimizer::updateStateVelocities(
  models::State & state) const
{
  updateInitialStateVelocities(state);
  propagateStateVelocitiesFromInitials(state);
}

void Optimizer::updateInitialStateVelocities(
  models::State & state) const
{
  xt::noalias(xt::view(state.vx, xt::all(), 0)) = state.speed.linear.x;
  xt::noalias(xt::view(state.wz, xt::all(), 0)) = state.speed.angular.z;

  if (isHolonomic()) {
    xt::noalias(xt::view(state.vy, xt::all(), 0)) = state.speed.linear.y;
  }
}

void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & state) const
{
  motion_model_->predict(state);
}

void Optimizer::integrateStateVelocities(
  xt::xtensor<float, 2> & trajectory,
  const xt::xtensor<float, 2> & sequence) const
{
  float initial_yaw = tf2::getYaw(state_.pose.pose.orientation);

  const auto vx = xt::view(sequence, xt::all(), 0);
  const auto vy = xt::view(sequence, xt::all(), 2);
  const auto wz = xt::view(sequence, xt::all(), 1);

  auto traj_x = xt::view(trajectory, xt::all(), 0);
  auto traj_y = xt::view(trajectory, xt::all(), 1);
  auto traj_yaws = xt::view(trajectory, xt::all(), 2);

  xt::noalias(traj_yaws) = cumsum_1d(wz * settings_.model_dt) + initial_yaw;

  auto && yaw_cos = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());

  const auto yaw_offseted = xt::view(traj_yaws, xt::range(1, _));

  xt::noalias(xt::view(yaw_cos, 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::range(1, _))) = xt::cos(yaw_offseted);
  xt::noalias(xt::view(yaw_sin, xt::range(1, _))) = xt::sin(yaw_offseted);

  auto && dx = xt::eval(vx * yaw_cos);
  auto && dy = xt::eval(vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - vy * yaw_sin;
    dy = dy + vy * yaw_cos;
  }

  xt::noalias(traj_x) = state_.pose.pose.position.x + cumsum_1d(dx * settings_.model_dt);
  xt::noalias(traj_y) = state_.pose.pose.position.y + cumsum_1d(dy * settings_.model_dt);
}

void Optimizer::integrateStateVelocities(
  models::Trajectories & trajectories,
  const models::State & state) const
{
  const float initial_yaw = tf2::getYaw(state.pose.pose.orientation);

  xt::noalias(trajectories.yaws) =
    cumsum_2d(state.wz * settings_.model_dt, 1) + initial_yaw;

  const auto yaws_cutted = xt::view(trajectories.yaws, xt::all(), xt::range(0, -1));

  auto && yaw_cos = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  xt::noalias(xt::view(yaw_cos, xt::all(), 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, xt::all(), 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::all(), xt::range(1, _))) = xt::cos(yaws_cutted);
  xt::noalias(xt::view(yaw_sin, xt::all(), xt::range(1, _))) = xt::sin(yaws_cutted);

  auto && dx = xt::eval(state.vx * yaw_cos);
  auto && dy = xt::eval(state.vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - state.vy * yaw_sin;
    dy = dy + state.vy * yaw_cos;
  }

  xt::noalias(trajectories.x) = state.pose.pose.position.x +
    cumsum_2d(dx * settings_.model_dt, 1);
  xt::noalias(trajectories.y) = state.pose.pose.position.y +
    cumsum_2d(dy * settings_.model_dt, 1);
}

xt::xtensor<float, 2> Optimizer::getOptimizedTrajectory()
{
  auto && sequence =
    xt::xtensor<float, 2>::from_shape({settings_.time_steps, isHolonomic() ? 3u : 2u});
  auto && trajectories = xt::xtensor<float, 2>::from_shape({settings_.time_steps, 3});

  xt::noalias(xt::view(sequence, xt::all(), 0)) = control_sequence_.vx;
  xt::noalias(xt::view(sequence, xt::all(), 1)) = control_sequence_.wz;

  if (isHolonomic()) {
    xt::noalias(xt::view(sequence, xt::all(), 2)) = control_sequence_.vy;
  }

  integrateStateVelocities(trajectories, sequence);
  return std::move(trajectories);
}

void Optimizer::updateControlSequence()
{
  auto & s = settings_;
  auto bounded_noises_vx = state_.cvx - control_sequence_.vx;
  auto bounded_noises_wz = state_.cwz - control_sequence_.wz;
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.vx, 2) * xt::sum(
    xt::view(control_sequence_.vx, xt::newaxis(), xt::all()) * bounded_noises_vx, 1, immediate);
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.wz, 2) * xt::sum(
    xt::view(control_sequence_.wz, xt::newaxis(), xt::all()) * bounded_noises_wz, 1, immediate);

  if (isHolonomic()) {
    auto bounded_noises_vy = state_.cvy - control_sequence_.vy;
    xt::noalias(costs_) +=
      s.gamma / powf(s.sampling_std.vy, 2) * xt::sum(
      xt::view(control_sequence_.vy, xt::newaxis(), xt::all()) * bounded_noises_vy,
      1, immediate);
  }

  auto && costs_normalized = costs_ - xt::amin(costs_, immediate);
  auto && exponents = xt::eval(xt::exp(-1 / settings_.temperature * costs_normalized));
  auto && softmaxes = xt::eval(exponents / xt::sum(exponents, immediate));
  auto && softmaxes_extened = xt::eval(xt::view(softmaxes, xt::all(), xt::newaxis()));

  xt::noalias(control_sequence_.vx) = xt::sum(state_.cvx * softmaxes_extened, 0, immediate);
  xt::noalias(control_sequence_.wz) = xt::sum(state_.cwz * softmaxes_extened, 0, immediate);
  if (isHolonomic()) {
    xt::noalias(control_sequence_.vy) = xt::sum(state_.cvy * softmaxes_extened, 0, immediate);
  }

  applyControlSequenceConstraints();
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp)
{
  unsigned int offset = settings_.shift_control_sequence ? 1 : 0;

  auto vx = control_sequence_.vx(offset);
  auto wz = control_sequence_.wz(offset);

  if (isHolonomic()) {
    auto vy = control_sequence_.vy(offset);
    return utils::toTwistStamped(vx, vy, wz, stamp, costmap_ros_->getBaseFrameID());
  }

  return utils::toTwistStamped(vx, wz, stamp, costmap_ros_->getBaseFrameID());
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_shared<OmniMotionModel>();
  } else if (model == "Ackermann") {
    motion_model_ = std::make_shared<AckermannMotionModel>(parameters_handler_, name_);
  } else {
    throw std::runtime_error(
            std::string(
              "Model " + model + " is not valid! Valid options are DiffDrive, Omni, "
              "or Ackermann"));
  }
}

void Optimizer::setSpeedLimit(double speed_limit, bool percentage)
{
  auto & s = settings_;
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    s.constraints.vx_max = s.base_constraints.vx_max;
    s.constraints.vx_min = s.base_constraints.vx_min;
    s.constraints.vy = s.base_constraints.vy;
    s.constraints.wz = s.base_constraints.wz;
  } else {
    if (percentage) {
      // Speed limit is expressed in % from maximum speed of robot
      double ratio = speed_limit / 100.0;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    } else {
      // Speed limit is expressed in absolute value
      double ratio = speed_limit / s.base_constraints.vx_max;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    }
  }
}

models::Trajectories & Optimizer::getGeneratedTrajectories()
{
  return generated_trajectories_;
}

}  // namespace amoeba

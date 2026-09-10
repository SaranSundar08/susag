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

#ifndef NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_
#define NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_

#include <string>
#include <memory>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_core/goal_checker.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "nav2_tgmppi_controller/models/optimizer_settings.hpp"
#include "nav2_tgmppi_controller/motion_models.hpp"
#include "nav2_tgmppi_controller/critic_manager.hpp"
#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"
#include "nav2_tgmppi_controller/tools/flow_field.hpp"
#include "nav2_tgmppi_controller/tools/noise_generator.hpp"
#include "nav2_tgmppi_controller/tools/parameters_handler.hpp"
#include "nav2_tgmppi_controller/tools/utils.hpp"

#ifdef __APPLE__
  #include "nav2_tgmppi_controller/tools/apple_utils.hpp"
#endif

namespace tgmppi
{

/**
 * @class tgmppi::Optimizer
 * @brief Main algorithm optimizer of the MPPI Controller
 */
class Optimizer
{
public:
  /**
    * @brief Constructor for tgmppi::Optimizer
    */
  Optimizer() = default;

  /**
   * @brief Destructor for tgmppi::Optimizer
   */
  ~Optimizer() {shutdown();}


  /**
   * @brief Initializes optimizer on startup
   * @param parent WeakPtr to node
   * @param name Name of plugin
   * @param costmap_ros Costmap2DROS object of environment
   * @param dynamic_parameter_handler Parameter handler object
   */
  void initialize(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    ParametersHandler * dynamic_parameters_handler);

  /**
   * @brief Shutdown for optimizer at process end
   */
  void shutdown();

  /**
   * @brief Compute control using MPPI algorithm
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   * @return TwistStamped of the MPPI control
   */
  geometry_msgs::msg::TwistStamped evalControl(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed, const nav_msgs::msg::Path & plan,
    nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Get the trajectories generated in a cycle for visualization
   * @return Set of trajectories evaluated in cycle
   */
  models::Trajectories & getGeneratedTrajectories();

  /**
   * @brief Get the optimal trajectory for a cycle for visualization
   * @return Optimal trajectory
   */
  xt::xtensor<float, 2> getOptimizedTrajectory();

  /**
   * @brief Set the maximum speed based on the speed limits callback
   * @param speed_limit Limit of the speed for use
   * @param percentage Whether the speed limit is absolute or relative
   */
  void setSpeedLimit(double speed_limit, bool percentage);

  /**
   * @brief Reset the optimization problem to initial conditions
   */
  void reset();

protected:
  /**
   * @brief Main function to generate, score, and return trajectories
   */
  void optimize();

  /**
   * @brief Prepare state information on new request for trajectory rollouts
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   */
  void prepare(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Obtain the main controller's parameters
   */
  void getParams();

  /**
   * @brief Set the motion model of the vehicle platform
   * @param model Model string to use
   */
  void setMotionModel(const std::string & model);

  /**
   * @brief Shift the optimal control sequence after processing for
   * next iterations initial conditions after execution
   */
  void shiftControlSequence();

  /**
   * @brief updates generated trajectories with noised trajectories
   * from the last cycle's optimal control
   */
  void generateNoisedTrajectories();

  /**
   * @brief Flow-mode bias: re-flood the water field
   * from the local costmap (every flow_reflood_every cycles), roll the
   * nominal control sequence out once, and at each horizon point shift the
   * sampling mean's yaw rate toward the field's downhill direction — the
   * whole horizon bends into the water channel, not just the first step,
   * and not just line-of-sight like the ray scan. The nominal
   * control_sequence_ is untouched so the gamma control-cost term supplies
   * the importance correction, same as the ray mode.
   */
  void applyFlowBias();

  /** @brief Add a constant global-path rejoin prior to each pseudopod row group. */
  void applyTgMppiModePriors();

  /** @brief True when the upcoming transformed global path is obstructed in
   * the local costmap. Used to keep TgMppi dormant during normal tracking. */
  bool isLocalPathBlocked() const;

  /**
   * @brief Publish the water field as RViz markers on /tgmppi_debug:
   * downhill direction arrows on a coarse subsample of wet cells plus a
   * state label with the re-flood time.
   */
  void publishFlowDebug();

  /**
   * @brief Publish up to three geodesic pseudopod candidates as lightweight
   * nav_msgs/Path topics, independently of dense MarkerArray visualization.
   */
  void publishAncillaryPaths();

  /**
   * @brief Publish the dynamically rolled-out, collision-validated ancillary
   * means. Empty paths clear rejected or inactive modes in RViz.
   */
  void publishAncillaryRollouts();

  /**
   * @brief Apply hard vehicle constraints on control sequence
   */
  void applyControlSequenceConstraints();

  /**
   * @brief  Update velocities in state
   * @param state fill state with velocities on each step
   */
  void updateStateVelocities(models::State & state) const;

  /**
   * @brief  Update initial velocity in state
   * @param state fill state
   */
  void updateInitialStateVelocities(models::State & state) const;

  /**
   * @brief predict velocities in state using model
   * for time horizon equal to timesteps
   * @param state fill state
   */
  void propagateStateVelocitiesFromInitials(models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    models::Trajectories & trajectories,
    const models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    xt::xtensor<float, 2> & trajectories,
    const xt::xtensor<float, 2> & state) const;

  /**
   * @brief Update control sequence with state controls weighted by costs
   * using softmax function
   */
  void updateControlSequence();

  /**
   * @brief Convert control sequence to a twist commant
   * @param stamp Timestamp to use
   * @return TwistStamped of command to send to robot base
   */
  geometry_msgs::msg::TwistStamped
  getControlFromSequenceAsTwist(const builtin_interfaces::msg::Time & stamp);

  /**
   * @brief Whether the motion model is holonomic
   * @return Bool if holonomic to populate `y` axis of state
   */
  bool isHolonomic() const;

  /**
   * @brief Using control frequence and time step size, determine if trajectory
   * offset should be used to populate initial state of the next cycle
   */
  void setOffset(double controller_frequency);

  /**
   * @brief Perform fallback behavior to try to recover from a set of trajectories in collision
   * @param fail Whether the system failed to recover from
   */
  bool fallback(bool fail);

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  std::string name_;

  std::shared_ptr<MotionModel> motion_model_;

  ParametersHandler * parameters_handler_;
  CriticManager critic_manager_;
  NoiseGenerator noise_generator_;

  models::OptimizerSettings settings_;

  models::State state_;
  models::ControlSequence control_sequence_;
  // Flow mode: the water field over the local costmap + re-flood cycle count.
  FlowField flow_field_;
  unsigned int flow_cycle_{0};
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    tgmppi_debug_pub_;
  std::array<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr, 3>
  ancillary_path_pubs_;
  std::array<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr, 3>
  ancillary_rollout_pubs_;
  std::array<std::vector<float>, 3> ancillary_rollout_x_;
  std::array<std::vector<float>, 3> ancillary_rollout_y_;
  std::array<bool, 3> ancillary_mode_valid_{{false, false, false}};
  std::array<unsigned int, 3> ancillary_mode_samples_{{0u, 0u, 0u}};
  std::array<unsigned int, 3> ancillary_mode_row_start_{{0u, 0u, 0u}};
  std::array<float, 3> ancillary_mode_rejoin_prior_{{0.0f, 0.0f, 0.0f}};
  bool tgmppi_assist_active_{false};
  bool flow_path_blocked_now_{false};
  unsigned int flow_clear_cycles_{0u};
  unsigned int flow_wait_samples_{0u};
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>
  ancillary_collision_checker_{nullptr};
  std::array<tgmppi::models::Control, 4> control_history_;
  models::Trajectories generated_trajectories_;
  models::Path path_;
  xt::xtensor<float, 1> costs_;

  CriticData critics_data_ =
  {state_, generated_trajectories_, path_, costs_, settings_.model_dt, false, nullptr, nullptr,
    std::nullopt, std::nullopt, nullptr};  /// Caution, keep references

  rclcpp::Logger logger_{rclcpp::get_logger("TgMppiController")};
};

template<typename E>
inline auto cumsum_1d(const E & expression)
{
  #ifdef __APPLE__
  return utils::manual_cumsum_1d(expression);
  #else
  return xt::cumsum(expression, 0);
  #endif
}

template<typename E>
inline auto cumsum_2d(const E & expression, int axis)
{
 #ifdef __APPLE__
  return utils::manual_cumsum_2d(expression, axis);
 #else
  return xt::cumsum(expression, axis);
 #endif
}

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_

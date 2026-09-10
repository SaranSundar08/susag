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

#include <algorithm>
#include <memory>
#include "nav2_tgmppi_controller/tools/trajectory_visualizer.hpp"

namespace tgmppi
{

void TrajectoryVisualizer::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  const std::string & frame_id, ParametersHandler * parameters_handler)
{
  auto node = parent.lock();
  logger_ = node->get_logger();
  frame_id_ = frame_id;
  trajectories_publisher_ =
    node->create_publisher<visualization_msgs::msg::MarkerArray>("/trajectories", 1);
  transformed_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("transformed_global_plan", 1);
  parameters_handler_ = parameters_handler;

  auto getParam = parameters_handler->getParamGetter(name + ".TrajectoryVisualizer");

  getParam(trajectory_step_, "trajectory_step", 5);
  getParam(time_step_, "time_step", 3);
  getParam(publish_every_n_, "publish_every_n", 1);
  trajectory_step_ = std::max<std::size_t>(1, trajectory_step_);
  time_step_ = std::max<std::size_t>(1, time_step_);
  publish_every_n_ = std::max<std::size_t>(1, publish_every_n_);

  reset();
}

void TrajectoryVisualizer::on_cleanup()
{
  trajectories_publisher_.reset();
  transformed_path_pub_.reset();
}

void TrajectoryVisualizer::on_activate()
{
  trajectories_publisher_->on_activate();
  transformed_path_pub_->on_activate();
}

void TrajectoryVisualizer::on_deactivate()
{
  trajectories_publisher_->on_deactivate();
  transformed_path_pub_->on_deactivate();
}

void TrajectoryVisualizer::add(
  const xt::xtensor<float, 2> & trajectory, const std::string & marker_namespace)
{
  auto & size = trajectory.shape()[0];
  if (!size) {
    return;
  }

  auto add_marker = [&](auto i) {
      float component = static_cast<float>(i) / static_cast<float>(size);

      auto pose = utils::createPose(trajectory(i, 0), trajectory(i, 1), 0.06);
      auto scale =
        i != size - 1 ?
        utils::createScale(0.03, 0.03, 0.07) :
        utils::createScale(0.07, 0.07, 0.09);
      auto color = utils::createColor(0, component, component, 1);
      auto marker = utils::createMarker(
        marker_id_++, pose, scale, color, frame_id_, marker_namespace);
      points_->markers.push_back(marker);
    };

  for (size_t i = 0; i < size; i++) {
    add_marker(i);
  }
}

void TrajectoryVisualizer::add(
  const models::Trajectories & trajectories, const std::string & marker_namespace)
{
  auto & shape = trajectories.x.shape();
  if (shape[0] == 0 || shape[1] < 2) {
    return;
  }
  const float shape_1 = static_cast<float>(shape[1]);

  // Batch the sampled trajectory cloud into one LINE_LIST marker. The old
  // representation created one RViz marker per point (thousands per update),
  // which dominated DDS serialization and RViz rendering time.
  visualization_msgs::msg::Marker lines;
  lines.header.frame_id = frame_id_;
  lines.header.stamp = rclcpp::Time(0, 0);
  lines.ns = marker_namespace;
  lines.id = marker_id_++;
  lines.type = visualization_msgs::msg::Marker::LINE_LIST;
  lines.action = visualization_msgs::msg::Marker::ADD;
  lines.pose.orientation.w = 1.0;
  lines.scale.x = 0.018;
  lines.color.a = 1.0;

  for (size_t i = 0; i < shape[0]; i += trajectory_step_) {
    for (size_t j = 0; j + time_step_ < shape[1]; j += time_step_) {
      const size_t next = j + time_step_;
      const float j_flt = static_cast<float>(j);
      float blue_component = 1.0f - j_flt / shape_1;
      float green_component = j_flt / shape_1;
      geometry_msgs::msg::Point p0;
      p0.x = trajectories.x(i, j);
      p0.y = trajectories.y(i, j);
      p0.z = 0.03;
      geometry_msgs::msg::Point p1;
      p1.x = trajectories.x(i, next);
      p1.y = trajectories.y(i, next);
      p1.z = 0.03;
      const auto color = utils::createColor(0, green_component, blue_component, 0.72);
      lines.points.push_back(p0);
      lines.points.push_back(p1);
      lines.colors.push_back(color);
      lines.colors.push_back(color);
    }
  }
  points_->markers.push_back(std::move(lines));
}

void TrajectoryVisualizer::reset()
{
  marker_id_ = 0;
  points_ = std::make_unique<visualization_msgs::msg::MarkerArray>();
}

void TrajectoryVisualizer::visualize(const nav_msgs::msg::Path & plan)
{
  const bool publish_this_cycle =
    visualization_cycle_++ % publish_every_n_ == 0;
  if (publish_this_cycle && trajectories_publisher_->get_subscription_count() > 0) {
    trajectories_publisher_->publish(std::move(points_));
  }

  reset();

  if (publish_this_cycle && transformed_path_pub_->get_subscription_count() > 0) {
    auto plan_ptr = std::make_unique<nav_msgs::msg::Path>(plan);
    transformed_path_pub_->publish(std::move(plan_ptr));
  }
}

}  // namespace tgmppi

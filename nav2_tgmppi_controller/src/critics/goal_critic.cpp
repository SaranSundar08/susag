// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
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

#include "nav2_tgmppi_controller/critics/goal_critic.hpp"
#ifdef TGMPPI_WITH_CUDA
#include <cstring>
#include "nav2_tgmppi_controller/tools/gpu_rollout.hpp"
#endif

namespace tgmppi::critics
{

using xt::evaluation_strategy::immediate;

void GoalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0);
  getParam(threshold_to_consider_, "threshold_to_consider", 1.4);

  RCLCPP_INFO(
    logger_, "GoalCritic instantiated with %d power and %f weight.",
    power_, weight_);

#ifdef TGMPPI_WITH_CUDA
  gpu_critic_.initialize();
#endif
}

void GoalCritic::score(CriticData & data)
{
  if (!enabled_ || !utils::withinPositionGoalTolerance(
      threshold_to_consider_, data.state.pose.pose, data.path))
  {
    return;
  }

  const auto goal_idx = data.path.x.shape(0) - 1;

  const auto goal_x = data.path.x(goal_idx);
  const auto goal_y = data.path.y(goal_idx);

#ifdef TGMPPI_WITH_CUDA
  if (data.compute_backend == "cuda" && gpu_critic_.ready()) {
    auto cost_out = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    if (data.gpu_rollout != nullptr) {
      // Chained: reads the rollout's already-resident trajectory tensors
      // directly. No upload of our own at all -- goal_x/goal_y are two
      // scalar kernel arguments, not tensors -- just a [K] download at
      // the end. The cheapest possible addition to the shared
      // data.gpu_rollout pipeline (see PROJECT_STATUS.md 2026-09-10).
      const auto * rollout = static_cast<const GpuRollout *>(data.gpu_rollout);
      auto cost_gpu = gpu_critic_.computeDevice(
        rollout->trajX(), rollout->trajY(), goal_x, goal_y, weight_, power_);
      auto cost_cpu = cost_gpu.to(torch::kCPU).contiguous();
      std::memcpy(
        cost_out.data(), cost_cpu.data_ptr<float>(), cost_out.size() * sizeof(float));
    } else {
      gpu_critic_.score(data.trajectories, goal_x, goal_y, weight_, power_, cost_out);
    }
    data.costs += cost_out;
    return;
  }
#endif

  const auto traj_x = xt::view(data.trajectories.x, xt::all(), xt::all());
  const auto traj_y = xt::view(data.trajectories.y, xt::all(), xt::all());

  auto dists = xt::sqrt(
    xt::pow(traj_x - goal_x, 2) +
    xt::pow(traj_y - goal_y, 2));

  data.costs += xt::pow(xt::mean(dists, {1}, immediate) * weight_, power_);
}

}  // namespace tgmppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(tgmppi::critics::GoalCritic, tgmppi::critics::CriticFunction)

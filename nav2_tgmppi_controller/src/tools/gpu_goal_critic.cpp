// Copyright (c) 2026 SLIP thesis fork
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

#include "nav2_tgmppi_controller/tools/gpu_goal_critic.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cstring>

namespace tgmppi
{

torch::Tensor GpuGoalCritic::computeDevice(
  const torch::Tensor & traj_x, const torch::Tensor & traj_y,
  double goal_x, double goal_y, float weight, unsigned int power)
{
  auto dx = traj_x - static_cast<float>(goal_x);
  auto dy = traj_y - static_cast<float>(goal_y);
  auto dists = torch::sqrt(dx * dx + dy * dy);           // [K,T]
  auto mean_dist = dists.mean(1);                        // [K]
  return torch::pow(mean_dist * weight, static_cast<double>(power));
}

void GpuGoalCritic::score(
  const models::Trajectories & trajectories,
  double goal_x, double goal_y, float weight, unsigned int power,
  xt::xtensor<float, 1> & cost_out)
{
  const int64_t K = static_cast<int64_t>(trajectories.x.shape(0));
  const int64_t T = static_cast<int64_t>(trajectories.x.shape(1));
  const std::vector<int64_t> shape{K, T};
  const auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);

  auto x = torch::from_blob(
    const_cast<float *>(trajectories.x.data()), shape, cpu_opts).to(torch::kCUDA);
  auto y = torch::from_blob(
    const_cast<float *>(trajectories.y.data()), shape, cpu_opts).to(torch::kCUDA);

  auto cost = computeDevice(x, y, goal_x, goal_y, weight, power);
  auto cost_cpu = cost.to(torch::kCPU).contiguous();
  std::memcpy(cost_out.data(), cost_cpu.data_ptr<float>(), static_cast<size_t>(K) * sizeof(float));
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

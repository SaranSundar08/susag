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

#include "nav2_tgmppi_controller/tools/gpu_elementwise_critics.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cstring>

namespace tgmppi
{

namespace
{
torch::Tensor uploadKT(const xt::xtensor<float, 2> & src)
{
  const int64_t K = static_cast<int64_t>(src.shape(0));
  const int64_t T = static_cast<int64_t>(src.shape(1));
  const auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::from_blob(
    const_cast<float *>(src.data()), {K, T}, cpu_opts).to(torch::kCUDA);
}

void downloadK(const torch::Tensor & src, xt::xtensor<float, 1> & dst)
{
  auto cpu = src.to(torch::kCPU).contiguous();
  std::memcpy(dst.data(), cpu.data_ptr<float>(), static_cast<size_t>(src.numel()) * sizeof(float));
}

// shortest_angular_distance(from, to) = normalize_angles(to - from), i.e.
// atan2(sin(to-from), cos(to-from)) -- see utils.hpp.
torch::Tensor shortestAngularDistance(const torch::Tensor & from, const torch::Tensor & to)
{
  auto d = to - from;
  return torch::atan2(torch::sin(d), torch::cos(d));
}
}  // namespace

// --- ConstraintCritic ---

torch::Tensor GpuElementwiseCritics::constraintCriticDevice(
  const torch::Tensor & vx, const torch::Tensor & vy, float model_dt,
  float max_vel, float min_vel, float weight, unsigned int power)
{
  auto sgn = torch::where(vx > 0.0, torch::ones_like(vx), -torch::ones_like(vx));
  auto vel_total = sgn * torch::sqrt(vx * vx + vy * vy);
  auto out_of_max = (vel_total - max_vel).clamp_min(0.0);
  auto out_of_min = (min_vel - vel_total).clamp_min(0.0);
  auto cost = torch::pow(
    (out_of_max + out_of_min).sum(1) * model_dt * weight, static_cast<double>(power));
  return cost;
}

void GpuElementwiseCritics::constraintCriticScore(
  const models::State & state, float model_dt, float max_vel, float min_vel,
  float weight, unsigned int power, xt::xtensor<float, 1> & cost_out)
{
  auto vx = uploadKT(state.vx);
  auto vy = uploadKT(state.vy);
  downloadK(constraintCriticDevice(vx, vy, model_dt, max_vel, min_vel, weight, power), cost_out);
}

// --- GoalAngleCritic ---

torch::Tensor GpuElementwiseCritics::goalAngleCriticDevice(
  const torch::Tensor & traj_yaws, float goal_yaw, float weight, unsigned int power)
{
  auto ang = torch::abs(shortestAngularDistance(traj_yaws, torch::full_like(traj_yaws, goal_yaw)));
  return torch::pow(ang.mean(1) * weight, static_cast<double>(power));
}

void GpuElementwiseCritics::goalAngleCriticScore(
  const models::Trajectories & trajectories, float goal_yaw,
  float weight, unsigned int power, xt::xtensor<float, 1> & cost_out)
{
  auto yaws = uploadKT(trajectories.yaws);
  downloadK(goalAngleCriticDevice(yaws, goal_yaw, weight, power), cost_out);
}

// --- PathFollowCritic ---

torch::Tensor GpuElementwiseCritics::pathFollowCriticDevice(
  const torch::Tensor & traj_x, const torch::Tensor & traj_y,
  double path_x, double path_y, float weight, unsigned int power)
{
  using namespace torch::indexing;  // NOLINT
  const int64_t T = traj_x.size(1);
  auto last_x = traj_x.index({Slice(), T - 1});
  auto last_y = traj_y.index({Slice(), T - 1});
  auto dists = torch::sqrt(
    torch::pow(last_x - static_cast<float>(path_x), 2) +
    torch::pow(last_y - static_cast<float>(path_y), 2));
  return torch::pow(weight * dists, static_cast<double>(power));
}

void GpuElementwiseCritics::pathFollowCriticScore(
  const models::Trajectories & trajectories, double path_x, double path_y,
  float weight, unsigned int power, xt::xtensor<float, 1> & cost_out)
{
  auto x = uploadKT(trajectories.x);
  auto y = uploadKT(trajectories.y);
  downloadK(pathFollowCriticDevice(x, y, path_x, path_y, weight, power), cost_out);
}

// --- PathAngleCritic ---

torch::Tensor GpuElementwiseCritics::pathAngleCriticDevice(
  const torch::Tensor & traj_x, const torch::Tensor & traj_y, const torch::Tensor & traj_yaws,
  double goal_x, double goal_y, float weight, unsigned int power)
{
  auto yaws_between = torch::atan2(
    static_cast<float>(goal_y) - traj_y, static_cast<float>(goal_x) - traj_x);
  auto yaws = torch::abs(shortestAngularDistance(traj_yaws, yaws_between));
  return torch::pow(yaws.mean(1) * weight, static_cast<double>(power));
}

void GpuElementwiseCritics::pathAngleCriticScore(
  const models::Trajectories & trajectories, double goal_x, double goal_y,
  float weight, unsigned int power, xt::xtensor<float, 1> & cost_out)
{
  auto x = uploadKT(trajectories.x);
  auto y = uploadKT(trajectories.y);
  auto yaws = uploadKT(trajectories.yaws);
  downloadK(pathAngleCriticDevice(x, y, yaws, goal_x, goal_y, weight, power), cost_out);
}

// --- PreferForwardCritic ---

torch::Tensor GpuElementwiseCritics::preferForwardCriticDevice(
  const torch::Tensor & vx, float model_dt, float weight, unsigned int power)
{
  auto backward = (-vx).clamp_min(0.0);
  return torch::pow(backward.sum(1) * model_dt * weight, static_cast<double>(power));
}

void GpuElementwiseCritics::preferForwardCriticScore(
  const models::State & state, float model_dt, float weight, unsigned int power,
  xt::xtensor<float, 1> & cost_out)
{
  auto vx = uploadKT(state.vx);
  downloadK(preferForwardCriticDevice(vx, model_dt, weight, power), cost_out);
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

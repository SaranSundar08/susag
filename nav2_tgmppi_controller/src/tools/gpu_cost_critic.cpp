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

#include "nav2_tgmppi_controller/tools/gpu_cost_critic.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cstring>

#include "nav2_costmap_2d/cost_values.hpp"

namespace tgmppi
{

void GpuCostCritic::initialize()
{
  ready_ = torch::cuda::is_available();
}

torch::Tensor GpuCostCritic::uploadCostmap(const nav2_costmap_2d::Costmap2D & costmap)
{
  const int64_t size_x = static_cast<int64_t>(costmap.getSizeInCellsX());
  const int64_t size_y = static_cast<int64_t>(costmap.getSizeInCellsY());
  const auto cpu_opts_u8 = torch::TensorOptions().dtype(torch::kUInt8);

  // Row-major, size_y rows of size_x, same layout as Costmap2D's internal
  // buffer (index = my * size_x + mx), so a direct reshape is correct with
  // no transpose.
  auto grid_cpu = torch::from_blob(
    const_cast<unsigned char *>(costmap.getCharMap()), {size_y, size_x}, cpu_opts_u8);
  return grid_cpu.to(torch::kFloat32).to(torch::kCUDA);
}

void GpuCostCritic::score(
  const models::Trajectories & trajectories,
  const nav2_costmap_2d::Costmap2D & costmap,
  bool is_tracking_unknown,
  float critical_cost, float collision_cost, bool near_goal,
  xt::xtensor<float, 1> & repulsive_cost_out,
  bool & all_trajectories_collide_out)
{
  const int64_t size_x = static_cast<int64_t>(costmap.getSizeInCellsX());
  const int64_t size_y = static_cast<int64_t>(costmap.getSizeInCellsY());
  const double origin_x = costmap.getOriginX();
  const double origin_y = costmap.getOriginY();
  const double resolution = costmap.getResolution();

  auto grid = uploadCostmap(costmap);

  const int64_t K = static_cast<int64_t>(trajectories.x.shape(0));
  const int64_t T = static_cast<int64_t>(trajectories.x.shape(1));
  const std::vector<int64_t> shape{K, T};
  const auto cpu_opts_f32 = torch::TensorOptions().dtype(torch::kFloat32);

  auto x = torch::from_blob(
    const_cast<float *>(trajectories.x.data()), shape, cpu_opts_f32).to(torch::kCUDA);
  auto y = torch::from_blob(
    const_cast<float *>(trajectories.y.data()), shape, cpu_opts_f32).to(torch::kCUDA);

  auto repulsive = computeDevice(
    x, y, grid, size_x, size_y, origin_x, origin_y, resolution,
    is_tracking_unknown, critical_cost, collision_cost, near_goal,
    all_trajectories_collide_out);

  auto repulsive_cpu = repulsive.to(torch::kCPU).contiguous();
  std::memcpy(
    repulsive_cost_out.data(), repulsive_cpu.data_ptr<float>(),
    static_cast<size_t>(K) * sizeof(float));
}

torch::Tensor GpuCostCritic::computeDevice(
  const torch::Tensor & x, const torch::Tensor & y,
  const torch::Tensor & grid, int64_t size_x, int64_t size_y,
  double origin_x, double origin_y, double resolution,
  bool is_tracking_unknown, float critical_cost, float collision_cost, bool near_goal,
  bool & all_trajectories_collide_out)
{
  const int64_t K = x.size(0);
  const int64_t T = x.size(1);

  // worldToMap(), vectorized: truncate-toward-zero on a non-negative value
  // is floor, matching Costmap2D::worldToMap() exactly. Out-of-bounds
  // points (negative offset, or past size_x/size_y) are masked separately
  // and treated as NO_INFORMATION, same as costAtPose()'s early return.
  auto mx_f = (x - static_cast<float>(origin_x)) / static_cast<float>(resolution);
  auto my_f = (y - static_cast<float>(origin_y)) / static_cast<float>(resolution);
  auto mx = mx_f.floor().to(torch::kLong);
  auto my = my_f.floor().to(torch::kLong);
  auto in_bounds = (mx_f >= 0) & (my_f >= 0) & (mx < size_x) & (my < size_y);

  // Clamp purely so the gather below never reads out of the tensor's
  // allocated memory; in_bounds masks the (possibly garbage) result right
  // after, so clamped-but-invalid indices never affect the output.
  auto mx_safe = mx.clamp(0, size_x - 1);
  auto my_safe = my.clamp(0, size_y - 1);
  auto flat_idx = my_safe * size_x + mx_safe;
  auto cost = grid.flatten().index({flat_idx.flatten()}).reshape({K, T});
  cost = torch::where(
    in_bounds, cost,
    torch::full_like(cost, static_cast<float>(nav2_costmap_2d::NO_INFORMATION)));

  // inCollision(), circular mode only (consider_footprint_==false is the
  // caller's precondition for using this class at all): LETHAL and
  // INSCRIBED_INFLATED always collide; NO_INFORMATION collides only if
  // the costmap isn't tracking unknown space as free.
  constexpr float kLethal = static_cast<float>(nav2_costmap_2d::LETHAL_OBSTACLE);
  constexpr float kInscribed = static_cast<float>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  constexpr float kUnknown = static_cast<float>(nav2_costmap_2d::NO_INFORMATION);
  auto in_collision = (cost == kLethal) | (cost == kInscribed) |
    ((cost == kUnknown) & !is_tracking_unknown);
  auto free_space = cost < 1.0f;

  // Per-point term, valid (i.e. actually used) only on non-colliding rows
  // -- see the class docstring for why it's safe to compute this
  // unconditionally (including past a "would-have-broken" point) and let
  // the final where() below discard it for collided trajectories.
  torch::Tensor term;
  if (near_goal) {
    term = torch::where(cost >= kInscribed, torch::full_like(cost, critical_cost), cost * 0.0f);
  } else {
    term = torch::where(cost >= kInscribed, torch::full_like(cost, critical_cost), cost);
  }
  term = torch::where(free_space, torch::zeros_like(term), term);

  auto sum_term = term.sum(1);                    // [K]
  auto any_collision = in_collision.any(1);        // [K], bool

  auto repulsive = torch::where(
    any_collision, torch::full_like(sum_term, collision_cost), sum_term);

  all_trajectories_collide_out = any_collision.all().item<bool>();
  return repulsive;
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

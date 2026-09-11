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

#include "nav2_tgmppi_controller/tools/gpu_flow_field_critic.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cstring>

namespace tgmppi
{

namespace
{
// Must match flow_field.cpp's anonymous-namespace kDryPenalty exactly --
// see the class docstring in gpu_flow_field_critic.hpp.
constexpr float kDryPenalty = 2.0f;
}  // namespace

torch::Tensor GpuFlowFieldCritic::uploadGrid(const FlowField & ff)
{
  const int64_t size_x = ff.sizeX();
  const int64_t size_y = ff.sizeY();
  const auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);
  // distGrid() is row-major, index = j*nx+i -- a [size_y, size_x] reshape
  // is correct with no transpose, same as GpuCostCritic's costmap upload.
  auto grid_cpu = torch::from_blob(
    const_cast<float *>(ff.distGrid().data()), {size_y, size_x}, cpu_opts);
  return grid_cpu.to(torch::kCUDA);
}

torch::Tensor GpuFlowFieldCritic::computeDevice(
  const torch::Tensor & traj_x, const torch::Tensor & traj_y,
  const torch::Tensor & grid, int64_t size_x, int64_t size_y,
  double origin_x, double origin_y, double resolution, float dmax,
  float weight, float running_weight, unsigned int power)
{
  using namespace torch::indexing;  // NOLINT

  const int64_t T = traj_x.size(1);

  // worldToGridClamped(): round to nearest cell, then clamp -- FlowField
  // always returns a valid, in-bounds cell (no out-of-bounds rejection),
  // so no mask is needed here (simpler than CostCritic's gather).
  auto i = ((traj_x - static_cast<float>(origin_x)) / static_cast<float>(resolution))
    .round().to(torch::kLong).clamp(0, size_x - 1);
  auto j = ((traj_y - static_cast<float>(origin_y)) / static_cast<float>(resolution))
    .round().to(torch::kLong).clamp(0, size_y - 1);
  auto flat_idx = j * size_x + i;
  auto d_raw = grid.flatten().index({flat_idx.flatten()}).reshape(traj_x.sizes());

  // distAt(): dry/unreached cells (>= kWetLimit) score dmax + kDryPenalty
  // instead of their raw (huge sentinel) value.
  auto d = torch::where(
    d_raw < FlowField::kWetLimit, d_raw,
    torch::full_like(d_raw, dmax + kDryPenalty));

  auto sum_over_t = d.sum(1);                              // [K]
  auto final_level = d.index({Slice(), T - 1});             // [K]
  auto cost = weight * final_level + running_weight * (sum_over_t / static_cast<float>(T));
  if (power > 1u) {
    cost = torch::pow(cost, static_cast<double>(power));
  }
  return cost;
}

void GpuFlowFieldCritic::score(
  const models::Trajectories & trajectories, const FlowField & ff,
  float weight, float running_weight, unsigned int power,
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
  auto grid = uploadGrid(ff);

  auto cost = computeDevice(
    x, y, grid, ff.sizeX(), ff.sizeY(), ff.originX(), ff.originY(), ff.resolution(), ff.dmax(),
    weight, running_weight, power);

  auto cost_cpu = cost.to(torch::kCPU).contiguous();
  std::memcpy(cost_out.data(), cost_cpu.data_ptr<float>(), static_cast<size_t>(K) * sizeof(float));
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

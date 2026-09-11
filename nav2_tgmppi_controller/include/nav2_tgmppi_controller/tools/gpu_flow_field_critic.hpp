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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_FLOW_FIELD_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_FLOW_FIELD_CRITIC_HPP_

#ifdef TGMPPI_WITH_CUDA

#include <torch/torch.h>

#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/tools/flow_field.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuFlowFieldCritic
 * @brief LibTorch/CUDA implementation of FlowFieldCritic::score(): a
 * per-point gather against FlowField's distance grid, exactly the same
 * shape as GpuCostCritic's costmap gather (see PROJECT_STATUS.md
 * 2026-09-10/11) -- simpler, actually, since FlowField::distAt() always
 * clamps to the grid rather than treating out-of-bounds specially, so no
 * in-bounds mask is needed here.
 *
 * One float constant is duplicated by necessity: flow_field.cpp's
 * kDryPenalty (2.0f, added to dmax() for dry/unreached cells) is a
 * private anonymous-namespace constant, not part of FlowField's public
 * interface. If that value ever changes, this must change with it -- the
 * parity test is what would catch a drift, not the type system.
 *
 * distAt() rounds to the nearest cell (std::lround, ties away from zero);
 * this uses torch::round (ties to even, IEEE round-half-to-even). The two
 * conventions only disagree exactly on a .5 cell-boundary tie, which
 * continuous random world coordinates essentially never land on -- not
 * observed in the parity test, noted here for honesty rather than found.
 */
class GpuFlowFieldCritic
{
public:
  GpuFlowFieldCritic() = default;

  void initialize() {ready_ = torch::cuda::is_available();}
  bool ready() const {return ready_;}

  /** @brief Standalone convenience: uploads trajectories.x/y and the flow
   * field's grid itself. Kept for use without GpuRollout. */
  void score(
    const models::Trajectories & trajectories, const FlowField & ff,
    float weight, float running_weight, unsigned int power,
    xt::xtensor<float, 1> & cost_out);

  /** @brief Upload-only: the flow field's distance grid as a
   * [size_y,size_x] CUDA tensor. */
  static torch::Tensor uploadGrid(const FlowField & ff);

  /**
   * @brief Device-resident compute only: traj_x/traj_y and grid are
   * already CUDA tensors (e.g. GpuRollout::trajX()/trajY() and
   * uploadGrid()). Returns the [K] cost tensor, still on the GPU.
   */
  torch::Tensor computeDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y,
    const torch::Tensor & grid, int64_t size_x, int64_t size_y,
    double origin_x, double origin_y, double resolution, float dmax,
    float weight, float running_weight, unsigned int power);

private:
  bool ready_{false};
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_FLOW_FIELD_CRITIC_HPP_

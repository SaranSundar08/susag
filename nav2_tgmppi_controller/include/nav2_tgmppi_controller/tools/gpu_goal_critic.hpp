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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_GOAL_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_GOAL_CRITIC_HPP_

#ifdef TGMPPI_WITH_CUDA

#include <torch/torch.h>

#include "nav2_tgmppi_controller/models/trajectories.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuGoalCritic
 * @brief LibTorch/CUDA implementation of GoalCritic::score()'s formula:
 * cost[k] = pow(weight * mean_t(dist((traj.x,traj.y)[k,t], goal)), power).
 * Pure elementwise + one mean-reduction over already-rolled-out
 * trajectories -- no costmap, no per-point branching, the simplest of the
 * three critics ported so far. When chained after GpuRollout via
 * computeDevice(), this critic needs NO upload of its own at all (goal_x/
 * goal_y are two scalars passed directly as kernel arguments, not
 * tensors) -- only a small [K] download of the result, the cheapest
 * possible addition to the shared data.gpu_rollout pipeline (see
 * PROJECT_STATUS.md 2026-09-10 "consolidation" entries for why chaining
 * onto shared resident tensors, not each critic uploading for itself,
 * is what makes GPU dispatch viable here at all).
 */
class GpuGoalCritic
{
public:
  GpuGoalCritic() = default;

  void initialize() {ready_ = torch::cuda::is_available();}
  bool ready() const {return ready_;}

  /**
   * @brief Standalone convenience: uploads trajectories.x/y itself, no
   * chaining. Kept for use without GpuRollout (parity tests, or
   * compute_backend:"cuda" with rollout unavailable).
   */
  void score(
    const models::Trajectories & trajectories,
    double goal_x, double goal_y, float weight, unsigned int power,
    xt::xtensor<float, 1> & cost_out);

  /**
   * @brief Device-resident compute only: traj_x/traj_y are already CUDA
   * tensors (e.g. GpuRollout::trajX()/trajY()), no upload happens here.
   * Returns the [K] cost tensor, still on the GPU -- caller downloads.
   */
  torch::Tensor computeDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y,
    double goal_x, double goal_y, float weight, unsigned int power);

private:
  bool ready_{false};
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_GOAL_CRITIC_HPP_

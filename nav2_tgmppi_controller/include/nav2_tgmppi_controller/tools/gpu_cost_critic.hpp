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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_COST_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_COST_CRITIC_HPP_

// Compiles to nothing unless built with -DTGMPPI_WITH_CUDA=ON. See
// gpu_rollout.hpp for the full rationale; same pattern here.
#ifdef TGMPPI_WITH_CUDA

#include <torch/torch.h>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuCostCritic
 * @brief LibTorch/CUDA implementation of CostCritic::score()'s CIRCULAR
 * collision-check path only (consider_footprint:false, the active
 * campaign yaml setting). Footprint mode needs per-point polygon
 * rasterization (footprintCostAtPose), which does not vectorize the same
 * way and is NOT covered here -- CostCritic::score() checks
 * consider_footprint_ itself and only calls this when it's false,
 * falling back to the existing CPU path otherwise.
 *
 * Replaces the K x T double loop (per-point costmap lookup + branch) with
 * one batched gather against the whole costmap uploaded as a tensor, plus
 * elementwise/reduction ops. The early-break-on-collision in the CPU
 * version doesn't need reproducing: a colliding trajectory's
 * repulsive_cost is unconditionally overwritten with collision_cost
 * afterward regardless of what partial sum preceded the break, so summing
 * every point unconditionally and then overwriting collided rows produces
 * an identical result -- see the parity test before trusting this.
 */
class GpuCostCritic
{
public:
  GpuCostCritic() = default;

  /** @brief Checks CUDA availability. Call once, e.g. from initialize(). */
  void initialize();

  bool ready() const {return ready_;}

  /**
   * @brief Circular-mode cost scoring, matching CostCritic::score()'s
   * consider_footprint_==false branch exactly.
   * @param trajectories Rolled-out x/y/yaws [K,T] (yaws unused -- circular
   * checking, like the CPU costAtPose(), ignores orientation).
   * @param costmap The live local costmap, uploaded fresh each call (it's
   * a rolling window that changes every cycle; no cross-cycle caching).
   * @param is_tracking_unknown costmap_ros_->getLayeredCostmap()->isTrackingUnknown()
   * @param critical_cost,collision_cost,near_goal Same meaning as the CPU critic.
   * @param repulsive_cost_out Filled with the per-trajectory cost (same
   * xtensor the CPU path fills) -- the caller applies weight/power/fail_flag
   * identically regardless of which backend filled it.
   * @param all_trajectories_collide_out Set true iff every trajectory collided.
   */
  void score(
    const models::Trajectories & trajectories,
    const nav2_costmap_2d::Costmap2D & costmap,
    bool is_tracking_unknown,
    float critical_cost, float collision_cost, bool near_goal,
    xt::xtensor<float, 1> & repulsive_cost_out,
    bool & all_trajectories_collide_out);

  /** @brief Upload-only: the local costmap as a [size_y,size_x] CUDA
   * tensor. A separate step (not folded into computeDevice()) so a caller
   * chaining multiple GPU critics against the same cycle's costmap only
   * uploads it once. */
  static torch::Tensor uploadCostmap(const nav2_costmap_2d::Costmap2D & costmap);

  /**
   * @brief Same computation as score(), but takes already-uploaded GPU
   * tensors (trajectories' x/y and the costmap) and returns the
   * repulsive-cost result resident on the GPU instead of downloading it --
   * for chaining after GpuRollout with no CPU round-trip in between. Used
   * by GpuCycle; score() is a thin upload-then-computeDevice-then-download
   * wrapper around this, kept for standalone use (the parity test, or
   * compute_backend:"cuda" with GpuRollout unavailable).
   */
  torch::Tensor computeDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y,
    const torch::Tensor & costmap_gpu, int64_t size_x, int64_t size_y,
    double origin_x, double origin_y, double resolution,
    bool is_tracking_unknown, float critical_cost, float collision_cost, bool near_goal,
    bool & all_trajectories_collide_out);

private:
  bool ready_{false};
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_COST_CRITIC_HPP_

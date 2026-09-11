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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ELEMENTWISE_CRITICS_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ELEMENTWISE_CRITICS_HPP_

#ifdef TGMPPI_WITH_CUDA

#include <torch/torch.h>

#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuElementwiseCritics
 * @brief The 5 remaining active critics (ConstraintCritic, GoalAngleCritic,
 * PathFollowCritic, PathAngleCritic, PreferForwardCritic) bundled into one
 * class: each is a single elementwise+reduction pass over already-rolled-
 * out state/trajectory tensors, structurally identical to GoalCritic
 * (see PROJECT_STATUS.md 2026-09-10) -- no persistent per-critic device
 * state, so one shared `ready_` flag covers all five instead of five
 * near-empty classes. Every method needs NO upload of its own when
 * chained onto GpuRollout's resident tensors (state.vx/vy/wz,
 * trajectories.x/y/yaws) -- goal points and scalar gates are computed on
 * CPU exactly as the originals do and passed in as plain arguments, not
 * tensors.
 *
 * Scoped to what the active campaign yaml actually exercises, matching
 * the CostCritic precedent (consider_footprint:false only): the
 * Ackermann branch of ConstraintCritic (motion_model is DiffDrive here),
 * GoalAngleCritic's symmetric_yaw_tolerance (unset -> false), and
 * PathAngleCritic's reversing-corrected branch (forward_preference
 * defaults true and is never overridden in the yaml, so it never fires)
 * are all NOT implemented here -- score() falls back to the CPU path
 * whenever a critic's own config would need one of those, so nothing
 * silently behaves differently if the yaml changes later.
 */
class GpuElementwiseCritics
{
public:
  void initialize() {ready_ = torch::cuda::is_available();}
  bool ready() const {return ready_;}

  // --- ConstraintCritic (non-Ackermann only) ---
  torch::Tensor constraintCriticDevice(
    const torch::Tensor & vx, const torch::Tensor & vy, float model_dt,
    float max_vel, float min_vel, float weight, unsigned int power);
  void constraintCriticScore(
    const models::State & state, float model_dt, float max_vel, float min_vel,
    float weight, unsigned int power, xt::xtensor<float, 1> & cost_out);

  // --- GoalAngleCritic (symmetric_yaw_tolerance:false only) ---
  torch::Tensor goalAngleCriticDevice(
    const torch::Tensor & traj_yaws, float goal_yaw, float weight, unsigned int power);
  void goalAngleCriticScore(
    const models::Trajectories & trajectories, float goal_yaw,
    float weight, unsigned int power, xt::xtensor<float, 1> & cost_out);

  // --- PathFollowCritic ---
  torch::Tensor pathFollowCriticDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y,
    double path_x, double path_y, float weight, unsigned int power);
  void pathFollowCriticScore(
    const models::Trajectories & trajectories, double path_x, double path_y,
    float weight, unsigned int power, xt::xtensor<float, 1> & cost_out);

  // --- PathAngleCritic (forward_preference:true only, no reversing correction) ---
  torch::Tensor pathAngleCriticDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y, const torch::Tensor & traj_yaws,
    double goal_x, double goal_y, float weight, unsigned int power);
  void pathAngleCriticScore(
    const models::Trajectories & trajectories, double goal_x, double goal_y,
    float weight, unsigned int power, xt::xtensor<float, 1> & cost_out);

  // --- PreferForwardCritic ---
  torch::Tensor preferForwardCriticDevice(
    const torch::Tensor & vx, float model_dt, float weight, unsigned int power);
  void preferForwardCriticScore(
    const models::State & state, float model_dt, float weight, unsigned int power,
    xt::xtensor<float, 1> & cost_out);

private:
  bool ready_{false};
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ELEMENTWISE_CRITICS_HPP_

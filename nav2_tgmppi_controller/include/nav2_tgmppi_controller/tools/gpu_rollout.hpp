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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ROLLOUT_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ROLLOUT_HPP_

// This entire file (and its .cpp) compiles to nothing unless the package was
// built with -DTGMPPI_WITH_CUDA=ON (a LibTorch install, e.g.
// ~/libtorch_cu126, is required only in that case). Default builds never
// see torch headers or need the dependency at all.
#ifdef TGMPPI_WITH_CUDA

#include <torch/torch.h>

#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuRollout
 * @brief LibTorch/CUDA implementation of Optimizer::updateStateVelocities()
 * + Optimizer::integrateStateVelocities() -- the diff-drive predict step
 * (one-timestep actuation-delay shift) and the cumulative kinematics
 * integration (semi-implicit Euler: position increment at t uses the yaw
 * from t-1). This is the K x T hot loop of MPPI, and the only piece of the
 * optimizer this first GPU slice covers -- noise generation and the
 * TG-MPPI flow bias stay on CPU (see optimizer.cpp's
 * generateNoisedTrajectories(): the bias mean-shift is small, serial,
 * per-mode xtensor logic that critics still need every cycle, so there is
 * nothing to gain from moving it before critics themselves move to GPU).
 *
 * Selected at runtime by the compute_backend:"cuda" param
 * (models::OptimizerSettings::compute_backend). Every downstream consumer
 * (critics, trajectory_visualizer) keeps reading the same CPU
 * models::State / models::Trajectories structs regardless of which
 * backend filled them -- only how vx/vy/wz and x/y/yaws got computed
 * changes, not what consumes them. Numerically verified against the
 * xtensor CPU path element-for-element; see the standalone parity test
 * referenced in PROJECT_STATUS.md before trusting this on the robot.
 */
class GpuRollout
{
public:
  GpuRollout() = default;

  /**
   * @brief (Re)allocate GPU tensors for the given batch/time-step shape.
   * Safe to call every Optimizer::reset() (batch_size/time_steps can change
   * via dynamic reconfigure). Sets ready() false if no CUDA device is
   * actually available at runtime, even if the binary was built with
   * TGMPPI_WITH_CUDA -- callers must fall back to the CPU path when that
   * happens, never assume compile-time support implies run-time support.
   */
  void initialize(unsigned int batch_size, unsigned int time_steps);

  /** @brief True once a CUDA device was found and tensors are allocated. */
  bool ready() const {return ready_;}

  /**
   * @brief Predict + integrate one full cycle on the GPU.
   * @param state_in Already noised (+ TG-MPPI bias-shifted) commanded
   * controls (cvx/cvy/cwz) and the robot's current pose/speed, read-only.
   * @param state_out Predicted velocities (vx/vy/wz) written here -- may be
   * the same object as state_in (only different fields are touched).
   * @param trajectories_out Rolled-out x/y/yaws written here.
   * @param model_dt Integration timestep.
   * @param is_holonomic Whether to also predict/integrate the y axis.
   */
  void rollout(
    const models::State & state_in, models::State & state_out,
    models::Trajectories & trajectories_out,
    float model_dt, bool is_holonomic);

  /**
   * @brief Same computation as rollout(), but takes already-uploaded GPU
   * tensors and leaves its results resident on the GPU (vx()/wz()/vy()/
   * trajX()/trajY()/trajYaws()) instead of downloading them -- for chaining
   * into another GPU stage (e.g. GpuCostCritic) with no CPU round-trip in
   * between. Used by GpuCycle; rollout() itself is now a thin
   * upload-then-computeDevice-then-download wrapper around this, kept for
   * standalone use (the parity test, or compute_backend:"cuda" with no
   * GPU-capable critics active).
   */
  void computeDevice(
    const torch::Tensor & cvx, const torch::Tensor & cwz, const torch::Tensor & cvy,
    float model_dt, float initial_yaw, double robot_x, double robot_y,
    float speed_vx, float speed_vy, float speed_wz, bool is_holonomic);

  const torch::Tensor & vx() const {return vx_;}
  const torch::Tensor & vy() const {return vy_;}
  const torch::Tensor & wz() const {return wz_;}
  const torch::Tensor & trajX() const {return traj_x_;}
  const torch::Tensor & trajY() const {return traj_y_;}
  const torch::Tensor & trajYaws() const {return traj_yaws_;}

private:
  bool ready_{false};
  unsigned int batch_size_{0};
  unsigned int time_steps_{0};
  torch::Tensor vx_, vy_, wz_;                 // predicted velocities [K,T]
  torch::Tensor traj_x_, traj_y_, traj_yaws_;  // rolled-out poses [K,T]
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_ROLLOUT_HPP_

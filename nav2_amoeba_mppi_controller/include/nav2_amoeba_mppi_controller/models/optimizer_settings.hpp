// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#ifndef NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
#define NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

#include <cstddef>
#include "nav2_amoeba_mppi_controller/models/constraints.hpp"

namespace amoeba::models
{

/**
 * @struct amoeba::models::OptimizerSettings
 * @brief Settings for the optimizer to use
 */
struct OptimizerSettings
{
  models::ControlConstraints base_constraints{0, 0, 0, 0};
  models::ControlConstraints constraints{0, 0, 0, 0};
  models::SamplingStd sampling_std{0, 0, 0};
  float model_dt{0};
  float temperature{0};
  float gamma{0};
  unsigned int batch_size{0};
  unsigned int time_steps{0};
  unsigned int iteration_count{0};
  bool shift_control_sequence{false};
  size_t retry_attempt_limit{0};

  // --- Amoeba shape-conditioned sampling (this controller's core behavior) ---
  float amoeba_bias_strength{0.6f};   // fraction of the batch seeded onto the wrap modes
  float amoeba_bias_gain{1.5f};       // heading P-gain: tangent bearing error -> yaw rate
  float amoeba_lookahead_dist{0.6f};  // path lookahead for the "toward-goal" bearing (m)
  float amoeba_scan_range{2.5f};      // forward ray-cast range into the local costmap (m)
  float amoeba_arc{1.57f};            // forward half-arc scanned each side (rad, ~90 deg)
  int   amoeba_num_rays{37};          // angular resolution of the polar gap-scan
  float amoeba_goal_dist{1.0f};       // within this range of the goal, disable the amoeba
                                      // (let plain MPPI + GoalCritic dock; kills endgame oscillation)
  bool  amoeba_debug{false};          // publish /amoeba_debug markers (scan rays + wrap arrows)
};

}  // namespace amoeba::models

#endif  // NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

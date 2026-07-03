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

#include "nav2_mppi_controller/tools/noise_generator.hpp"

#include <memory>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

namespace mppi
{

void NoiseGenerator::initialize(
  mppi::models::OptimizerSettings & settings, bool is_holonomic,
  const std::string & name, ParametersHandler * param_handler)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  active_ = true;

  auto getParam = param_handler->getParamGetter(name);
  getParam(regenerate_noises_, "regenerate_noises", false);

  if (regenerate_noises_) {
    noise_thread_ = std::thread(std::bind(&NoiseGenerator::noiseThread, this));
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::shutdown()
{
  active_ = false;
  ready_ = true;
  noise_cond_.notify_all();
  if (noise_thread_.joinable()) {
    noise_thread_.join();
  }
}

void NoiseGenerator::generateNextNoises()
{
  // Trigger the thread to run in parallel to this iteration
  // to generate the next iteration's noises (if applicable).
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    ready_ = true;
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::setNoisedControls(
  models::State & state,
  const models::ControlSequence & control_sequence)
{
  std::unique_lock<std::mutex> guard(noise_lock_);

  xt::noalias(state.cvx) = control_sequence.vx + noises_vx_;
  xt::noalias(state.cvy) = control_sequence.vy + noises_vy_;
  xt::noalias(state.cwz) = control_sequence.wz + noises_wz_;
}

void NoiseGenerator::reset(mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;

  // Recompute the noises on reset, initialization, and fallback
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    xt::noalias(noises_vx_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    xt::noalias(noises_vy_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    xt::noalias(noises_wz_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    ready_ = true;
  }

  if (regenerate_noises_) {
    noise_cond_.notify_all();
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::noiseThread()
{
  do {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noise_cond_.wait(guard, [this]() {return ready_;});
    ready_ = false;
    generateNoisedControls();
  } while (active_);
}

void NoiseGenerator::generateNoisedControls()
{
  // SLIP: dispatch to the selected MPPI variant. All variants only change how
  // the sampling noise is drawn; the rest of the pipeline is unchanged.
  switch (settings_.variant) {
    case models::MPPIVariant::LOG:
      generateLogNormalNoise();
      break;
    case models::MPPIVariant::LOWPASS:
      generateLowPassNoise();
      break;
    case models::MPPIVariant::BIASED:
      generateBiasedNoise();
      break;
    case models::MPPIVariant::VANILLA:
    default:
      generateGaussianNoise();
      break;
  }
}

void NoiseGenerator::generateGaussianNoise()
{
  // Vanilla MPPI: i.i.d. zero-mean Gaussian perturbations per (sample, timestep).
  auto & s = settings_;

  xt::noalias(noises_vx_) = xt::random::randn<float>(
    {s.batch_size, s.time_steps}, 0.0f,
    s.sampling_std.vx);
  xt::noalias(noises_wz_) = xt::random::randn<float>(
    {s.batch_size, s.time_steps}, 0.0f,
    s.sampling_std.wz);
  if (is_holonomic_) {
    xt::noalias(noises_vy_) = xt::random::randn<float>(
      {s.batch_size, s.time_steps}, 0.0f,
      s.sampling_std.vy);
  }
}

void NoiseGenerator::generateLogNormalNoise()
{
  // log-MPPI (Mohamed et al.): multiply each Gaussian sample by a log-normal
  // scalar so the injected noise has heavier tails -> escapes local minima in
  // cluttered/tight spaces. scale = exp(mu + sigma * N(0,1)); mu = -sigma^2/2
  // keeps E[scale] = 1 so the nominal sampling magnitude is preserved.
  auto & s = settings_;
  const float sigma = s.lognormal_sigma;
  const float mu = -0.5f * sigma * sigma;
  const std::vector<std::size_t> shape = {s.batch_size, s.time_steps};

  xt::xtensor<float, 2> scale =
    xt::exp(mu + sigma * xt::random::randn<float>(shape, 0.0f, 1.0f));
  xt::noalias(noises_vx_) =
    xt::random::randn<float>(shape, 0.0f, s.sampling_std.vx) * scale;

  scale = xt::exp(mu + sigma * xt::random::randn<float>(shape, 0.0f, 1.0f));
  xt::noalias(noises_wz_) =
    xt::random::randn<float>(shape, 0.0f, s.sampling_std.wz) * scale;

  if (is_holonomic_) {
    scale = xt::exp(mu + sigma * xt::random::randn<float>(shape, 0.0f, 1.0f));
    xt::noalias(noises_vy_) =
      xt::random::randn<float>(shape, 0.0f, s.sampling_std.vy) * scale;
  }
}

void NoiseGenerator::generateLowPassNoise()
{
  // LP-MPPI: draw white Gaussian noise, then low-pass filter it ALONG THE TIME
  // AXIS (causal exponential moving average across columns) so each sampled
  // control trajectory is smooth and dynamically feasible. This filters the
  // INPUT noise -- distinct from the Savitzky-Golay OUTPUT smoother in
  // Optimizer::evalControl(): one shapes the search, the other polishes the
  // committed command.
  //
  // Knobs (yaml): lowpass_cutoff_hz  = -3dB cutoff, lower = smoother
  //               lowpass_order      = cascaded EMA passes, higher = steeper rolloff
  auto & s = settings_;
  generateGaussianNoise();

  // Cutoff is specified in Hz and converted to the EMA coefficient using
  // model_dt, so it is invariant to the control rate:
  //   fc = -ln(1-a) / (2*pi*dt)  =>  a = 1 - exp(-2*pi*fc*dt)
  const float a =
    1.0f - std::exp(-2.0f * static_cast<float>(M_PI) * s.lowpass_cutoff_hz * s.model_dt);
  const int order = std::max(1, s.lowpass_order);

  // Cascade `order` causal EMA passes; each pass adds ~-6 dB/octave rolloff.
  auto lowpass = [&](xt::xtensor<float, 2> & n) {
      for (int pass = 0; pass < order; ++pass) {
        for (std::size_t t = 1; t < s.time_steps; ++t) {
          xt::view(n, xt::all(), t) =
            a * xt::view(n, xt::all(), t) +
            (1.0f - a) * xt::view(n, xt::all(), t - 1);
        }
      }
    };

  // EMA shrinks variance; rescale so the per-step marginal std matches
  // sampling_std. This isolates the temporal correlation as the ONLY difference
  // vs vanilla (not a hidden drop in exploration magnitude). Empirical RMS
  // normalization is exact for any order and warm-up transient.
  auto renormalize = [](xt::xtensor<float, 2> & n, float target_std) {
      const float rms = std::sqrt(xt::mean(n * n)());
      if (rms > 1e-9f) {
        n *= target_std / rms;
      }
    };

  lowpass(noises_vx_);
  renormalize(noises_vx_, s.sampling_std.vx);
  lowpass(noises_wz_);
  renormalize(noises_wz_, s.sampling_std.wz);
  if (is_holonomic_) {
    lowpass(noises_vy_);
    renormalize(noises_vy_, s.sampling_std.vy);
  }
}

void NoiseGenerator::generateBiasedNoise()
{
  // Biased-MPPI (Trevisan & Alonso-Mora): the perturbation noise itself stays
  // zero-mean Gaussian. The "bias" is a MEAN SHIFT applied in the Optimizer
  // (Optimizer::applyBiasedControls) -- a fraction of samples are re-centered
  // onto an ancillary controller's command -- and the importance-weight
  // correction is the existing gamma control-cost term in updateControlSequence.
  // So here we only need vanilla Gaussian noise.
  generateGaussianNoise();
}

}  // namespace mppi

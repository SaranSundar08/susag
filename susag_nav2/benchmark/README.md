# TgMppi vs vanilla MPPI benchmark

Paired A/B harness. It drives a fixed start→goal repeatedly, spawning a
different obstacle each scenario, and logs one CSV row per trial. Run it once
per controller (you swap the controller by launching nav2 with a different
params file); both runs append to the same `benchmark_results.csv`.

## 0. Tune the course FIRST (required)

Open `benchmark_tgmppi.py` and set `START` / `GOAL` to a **straight,
obstacle-free corridor** in your map (`maze_map_new`: x ∈ [-8.51, 9.99],
y ∈ [-9.85, 8.15]). The default is (0,0)→(4,0); confirm in RViz that the
global path between them is a clean straight line with free space either side,
otherwise the spawned obstacle isn't the only thing in the way and the result
is meaningless. The obstacles are auto-placed on the midpoint of that segment.

## 1. Vanilla run

```bash
# terminal A – sim + nav2 with the VANILLA params
ros2 launch susag_nav2 navigation.launch.py sim:=true \
     nav2_params:=$(ros2 pkg prefix susag_nav2)/share/susag_nav2/param/navigation_sim.yaml
# In RViz, give a 2D Pose Estimate so AMCL is localized (map frame appears).
```

```bash
# terminal B
cd src/susag_nav2/benchmark
python3 benchmark_tgmppi.py --controller vanilla --reps 10
```

## 2. TgMppi run

Ctrl-C nav2 in terminal A, relaunch with the tgmppi params:

```bash
ros2 launch susag_nav2 navigation.launch.py sim:=true \
     nav2_params:=$(ros2 pkg prefix susag_nav2)/share/susag_nav2/param/navigation_tgmppi.yaml
# 2D Pose Estimate again.
python3 benchmark_tgmppi.py --controller tgmppi --reps 10
```

## 3. Compare

```bash
python3 benchmark_tgmppi.py --report
```

Prints success-rate / time / path / clearance / stall per (scenario × controller).

## What to read

- **Headline = `succ%`.** Expect S0 ≈ tie; S1–S3 tgmppi higher; S4 tgmppi
  recovers where vanilla wanders. `time_s` is secondary (successful runs only)
  and vanilla will often win it on S0 — that's expected, not a loss.
- **`stall_s`** is the mode-averaging proxy: seconds spent creeping (|v|<0.03)
  while still far from goal. Vanilla should stall hard on S1/S3 if the trap bites.
- `min_clear_m` is nearest-anything from /scan (includes maze walls), so read it
  as a safety floor, not obstacle-specific distance.

## Quick single-scenario smoke test

```bash
python3 benchmark_tgmppi.py --controller tgmppi --reps 2 --scenarios S1_pillar
```

## Notes / limits

- Resets by **navigating home** (empty world has no `set_entity_state`), so a
  failed trial that leaves the robot stuck costs one extra home leg.
- No control-rate logging here; check that separately with
  `... | grep -i "missed its desired rate"` on the nav2 launch output.
- Uses the `navigate_to_pose` action and `map`-frame goals, so AMCL must be
  localized before you start (step 1's 2D Pose Estimate).

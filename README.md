# cognav_planner

ROS 2 (Humble) implementation of CogNav, a behaviour tree that explores indoor
environments from a single monocular camera. Depth Anything V2 turns each RGB
frame into a polar distance vector over the camera's field of view. An RRT
planner grows a tree inside the visible free space and commits to the branch
that reaches the least visited ground. A follow-the-gap driver proposes a
velocity command, and a swept-arc safety gate checks that exact command before
it is published. The robot keeps no map, only a short trail of past poses.

The repository contains the stack, the Habitat-Sim and TurtleBot4 platform
adapters, the episode manifests and the scripts that run and score the
experiments.

## Packages

| Package | Contents |
|---|---|
| `cognav_bt_behaviors` | Tree leaves (planner, driver, safety gate, conditions), their blackboard contracts, runtime parameters and the freshness deadline |
| `cognav_bt_runner` | Node that loads a mission XML and ticks it once per perception frame; tick log; launch file |
| `cognav_mission` | `ExecuteMission` action, the mission node and the mission files |
| `cognav_representation` | Geometry on the polar vector, the free-space oracles and the navmesh raster |
| `cognav_perception` | Perception node: Depth Anything V2 to polar distance vector, depth-scale calibration tools |
| `cognav_platform_adapter` | Habitat-Sim server and bridge, TurtleBot4 topic relay, laboratory scoring topics |
| `cognav_evaluation` | Episode manifests, tick-log parsing, metrics, paired comparison, scene descriptors |
| `cognav_msgs`, `cognav_visualization` | `DistanceVector` message; RViz markers |

## Experiment arms

The four tree arms differ only in the mission file. `mesh` runs the full
mission with the planner's free-space test answered by the scene's navmesh,
restricted to the same field of view and range. It is an upper reference for
this policy with exact geometry.

| Arm | Mission | Retry | Escape | Free space |
|---|---|---|---|---|
| `tree` | `random_explore.xml` | yes | yes | polar vector |
| `no_retry` | `no_retry.xml` | no | yes | polar vector |
| `no_escape` | `no_escape.xml` | yes | no | polar vector |
| `navigate_only` | `navigate_only.xml` | no | no | polar vector |
| `mesh` | `random_explore.xml` | yes | yes | navmesh |

`scripts/arm_dispatch.sh` defines the arms for every script.

## Requirements

- Docker with the NVIDIA Container Toolkit and a CUDA-capable GPU.
- Habitat scene data containing the scenes named in `episodes/*.json`
  (Gibson scenes, the HM3D v0.2 examples and the Habitat test scenes).

The image contains ROS 2 Humble, the workspace and two conda environments:
`perception` (Python 3.10, torch, Depth Anything V2) and `habitat`
(Python 3.9, habitat-sim). The metric Depth Anything V2 Small checkpoint is
downloaded during the build.

## Build

```bash
sudo ./scripts/build_image.sh
COGNAV_HABITAT_SRC=/path/to/habitat_data ./scripts/link_data.sh
```

`link_data.sh` links the scene data to `habitat_data/` in the checkout, which
the scripts mount into the container.

## Depth-scale calibration

`depth_scale` multiplies the raw Depth Anything V2 output and depends on the
scene. It is measured against Habitat's ground-truth depth:

```bash
sudo ./scripts/run_container.sh -d
./scripts/calibrate_depth_scale.sh hm3d_large
./scripts/calibrate_depth_scale.sh --hypersim logs/hypersim/ai_002_001   # control, expect ~1.0
```

The tool renders RGB and depth pairs from navigable poses, compares only the
pixels the encoder keeps, and prints the median ratio with its interquartile
range. Record the value in `src/cognav_perception/config/depth_scale.yaml` for
a scene alias, or in the scene's episode manifest. The manifests shipped here
already carry their values.

## Navmesh rasters for the mesh arm

```bash
docker exec -it cognav bash -lc 'cd /opt/ws && conda run -n habitat \
    python scripts/export_navmesh.py \
    --episodes episodes/{Albertville,Adrian,hm3d_medium,Bowlus,hm3d_large}.json \
    --out navmesh/'
```

This runs in the container started by `run_container.sh -d` and writes
`navmesh/<scene>.npz` in the checkout.

## One episode

```bash
sudo ./scripts/run_episode.sh --arm tree scene:=hm3d_large
sudo ./scripts/run_episode.sh --arm mesh --episode 0 scene:=Bowlus
./scripts/run_episode.sh --list-arms
```

Arguments after the flags go to `ros2 launch cognav_bt_runner cognav_bt_launch.py`.
The tick log is written to `logs/`. `./scripts/run_rviz.sh` opens RViz in the
running container.

## The batch

```bash
./scripts/run_experiments.sh --dry-run
sudo ./scripts/run_experiments.sh
sudo ./scripts/run_experiments.sh --scenes "Adrian Bowlus" --episodes 3 --arms "tree mesh"
```

By default: five scenes, the first ten episodes of each manifest and all five
arms, with a 500 s budget per episode. Every arm replays the same
(scene, start pose, seed) triples. Logs go to
`results/batch_<tag>/<arm>/<episode>.log`. Episodes that already have an end
marker are skipped, so an interrupted batch resumes with the same command.

## Scoring

```bash
conda run -n habitat python scripts/score_batch.py --batch results/batch_<tag>
```

Coverage is the observed area intersected with the navigable area of the
starting island, divided by that island's area, both counted on the same
lattice (`scripts/seen_area.py`). Results are printed per scene and arm; scenes
are not pooled. Further metrics and paired tests come from `cognav_evaluation`:

```bash
mkdir -p tree navigate_only
cognav_metrics results/batch_<tag>/tree/Adrian_*.log --json tree/Adrian.json
cognav_compare navigate_only/Adrian.json tree/Adrian.json     # Wilcoxon, McNemar
cognav_coverage_curve --arm tree=tree --arm navigate_only=navigate_only --scene Adrian --out figures/Adrian.pdf
cognav_arena --scenes <scene.glb>                             # area, occupancy, narrowness
```

## Real robot

On a TurtleBot4 (OAK-D camera), the same launch file runs with the robot
relay. With a `map_server` map of the laboratory it also starts AMCL and
`lab_bridge`, which publish the pose, clearance and bumper contacts used for
scoring. The stack itself drives on wheel odometry.

```bash
ros2 launch cognav_bt_runner cognav_bt_launch.py platform:=turtlebot \
    use_sim_time:=false robot_ns:=/<robot namespace> \
    lab_map:=/path/to/lab.yaml depth_scale:=<measured> enable_cmd_vel:=true
```

## Mission action

With `mission_autostart:=false` the mission node waits for a goal:

```bash
ros2 action send_goal /execute_mission cognav_mission/action/ExecuteMission \
    "{mission_xml_path: 'cognav_mission/mission/random_explore.xml', enable_cmd_vel: true}"
```

## Tests

The suites need a built workspace but no simulator, GPU or running ROS graph.

```bash
ws=/tmp/cognav_ws && mkdir -p $ws/src && ln -s $PWD/src/cognav_* $ws/src/
source /opt/ros/humble/setup.bash
(cd $ws && PATH=/usr/bin:$PATH colcon build --symlink-install \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3)
source $ws/install/setup.bash && ./tests/run_all.sh
```

The `PATH` and `Python3_EXECUTABLE` settings keep colcon on the system Python
when conda is active.

## License

Apache License 2.0, see [LICENSE](LICENSE).

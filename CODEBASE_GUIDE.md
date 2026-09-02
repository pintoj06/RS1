# Codebase Guide — 41068 Ignition Bringup

This is a plain-language reference for this package: what ROS 2 concepts you need,
what each file type is, and what every file in this repo actually does. Come back
to this any time you've forgotten what something does.

---

## Part 1 — ROS 2 concepts, explained simply

ROS 2 (Robot Operating System 2) is not an operating system — it's a framework for
getting many separate programs to talk to each other. Everything below is a concept
you'll see over and over in this codebase.

### Node
A **node** is one running program — e.g. "the SLAM node," "the camera driver node,"
"your autonomy script." A robot system is just lots of small nodes running at once
and talking to each other. `basic_autonomy_demo.py` is an example of a node.

### Topic
A **topic** is a named channel that nodes publish messages to, or subscribe to read
messages from. It's one-way, many-to-many, and continuous (a stream), like a radio
frequency. Example: `/husky1/scan` carries lidar data; any node can subscribe to it
without the publisher knowing or caring who's listening.

### Message type
Every topic carries one specific **message type** — a fixed data structure. E.g.
`sensor_msgs/msg/Image` is an image, `nav_msgs/msg/OccupancyGrid` is a 2D map,
`geometry_msgs/msg/Twist` is a velocity command (linear x/y/z + angular x/y/z).

### Service
A **service** is a request/response call — you ask a question, you get one answer
back, like a phone call. Used for occasional one-off asks rather than streams.
Gazebo's `set_pose` (used in `dynamic_world_demo.py`) is a service call.

### Action
An **action** is for long-running tasks with feedback and a final result — like
ordering food: you place an order (goal), get progress updates (feedback), and
eventually a result (done/failed). Nav2's `NavigateToPose` is an action: you send a
goal pose, get periodic "distance remaining" feedback, and eventually success/failure.
See this pattern in `basic_autonomy_demo.py`'s `_send_goal` / `_feedback_callback` /
`_result_callback`.

### Parameter
A **parameter** is a named setting a node reads at startup (or later), e.g.
`free_cell_threshold` in `basic_autonomy_demo.py`. Parameters usually come from a
YAML file or from the launch file that starts the node.

### Namespace
A **namespace** is a prefix applied to all of a node's topics/services/actions so
multiple copies of the same node type don't collide. This package puts the Husky
under `/husky1` and the Parrot under `/parrot1`, so `map` becomes `/husky1/map` and
`/parrot1/map` respectively — same code, no clashes, and it's what makes running two
robots at once possible without rewriting anything.

### TF (transform tree)
**TF** answers "where is X relative to Y?" for every part of the robot and the
world, continuously, as things move. Each robot has a chain of coordinate frames:
`<robot>_map` → `<robot>_odom` → `<robot>_base_link` → sensor frames like
`<robot>_base_scan`, `<robot>_camera_link`. `basic_autonomy_demo.py`'s
`_lookup_robot_pose()` uses TF to find "where is the robot right now, in map
coordinates?"

- `map` frame: fixed to the world, corrected by SLAM (best long-term global estimate).
- `odom` frame: drifts slowly over time, but is smooth (no sudden jumps) — good for
  short-term motion.
- `base_link`: the robot's own body frame.

### Launch file
A **launch file** (`.launch.py`) is a Python script that starts a group of nodes
together with the right parameters, remappings, and conditions — instead of you
manually running ten separate `ros2 run` commands in ten terminals.

### SLAM
**SLAM** (Simultaneous Localisation and Mapping) builds a map of an unknown
environment while simultaneously figuring out where the robot is inside that map,
using lidar/sensor data. This package uses the `slam_toolbox` package for this.

### Nav2
**Nav2** is ROS 2's navigation stack: given a map and a goal pose, it plans a path
and drives the robot there, avoiding obstacles, using an internal costmap built from
sensor data. You interact with it mainly through the `NavigateToPose` action.

### EKF / robot_localization
An **EKF** (Extended Kalman Filter) fuses multiple noisy sensor sources (e.g. wheel
odometry + IMU) into one smoother, more accurate pose estimate. The
`robot_localization` package's `ekf_node` does this here, publishing a filtered
`odom` topic.

---

## Part 2 — File type glossary

| Extension | What it is |
|---|---|
| `.py` (in `launch/`) | A launch file — starts and configures a group of nodes. |
| `.py` (in `scripts/`) | An actual ROS node — a running program with logic. |
| `.urdf.xacro` | Robot description: links (rigid parts), joints (how parts connect/move), sensor placements, visual meshes. `xacro` is a macro language that generates plain URDF — it lets you use variables/reuse (e.g. `${prefix}` so the same file works for `husky1` or a future `husky2`). |
| `.gazebo.xacro` | Gazebo-specific additions to a robot description: physics plugins, sensor simulation parameters (noise, update rate), which aren't part of the "real robot" description. |
| `.sdf` | Simulation Description Format — Gazebo's native format for describing worlds (`worlds/*.sdf`) or standalone models (`models/*/model.sdf`). Different from URDF; URDF describes *robots* for ROS generally, SDF describes *everything* for Gazebo specifically. |
| `.yaml` | Plain structured config data (key: value), used here for: Nav2/SLAM/EKF tuning parameters, and topic-bridge definitions (which Gazebo topics map to which ROS topics). |
| `.rviz` | A saved layout/config for RViz (the 3D visualisation tool) — which displays are shown, their settings, camera view. |
| `model.config` | Metadata for a Gazebo model folder (name, version, path to the `.sdf`) — lets Gazebo find and load a model by name. |

---

## Part 3 — How a simulation actually boots up

Running, e.g.:
```
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true
```
triggers this chain:

1. **`41068_ignition_husky.launch.py`** — a thin wrapper. It just calls the file
   below with `husky:=True, parrot:=False` baked in.
2. **`41068_ignition.launch.py`** (the canonical launch file) does the real work:
   - Starts Gazebo itself, loading the chosen world file.
   - Bridges the simulation clock to ROS (so all nodes agree on "sim time").
   - For each enabled robot, calls a local helper function `add_robot(...)` which:
     - Turns the robot's `.urdf.xacro` into a full robot description and publishes
       it (via `robot_state_publisher`, which also starts broadcasting TF).
     - Starts the EKF localisation node.
     - Spawns the robot entity into the running Gazebo world.
     - Starts a `parameter_bridge` node that mirrors the specific Gazebo topics
       listed in that robot's `gazebo_bridge_*.yaml` into ROS topics (and vice
       versa for `cmd_vel`).
   - If `slam` or `nav2` are true, includes **`41068_navigation.launch.py`**, which
     wraps the official `slam_toolbox` and `nav2_bringup` launch files, pushed into
     that robot's namespace.
   - If `rviz` is true, starts one RViz window per enabled robot, pointed at that
     robot's saved `.rviz` layout.

Everything after step 1 is namespaced (`husky1` or `parrot1`), which is why the same
launch code supports Husky-only, Parrot-only, or both at once — it's just whether
each robot's block of actions is enabled.

The two "add-on" launch files (`41068_autonomy_demo.launch.py`,
`41068_dynamic_world_demo.launch.py`) are separate and lightweight: they assume the
simulation above is already running, and just start one extra Python node each.

---

## Part 4 — File-by-file reference

### `launch/`
| File | Role |
|---|---|
| `41068_ignition.launch.py` | Canonical launch file — starts Gazebo + robots + (optionally) SLAM/Nav2/RViz. Everything else calls into this. |
| `41068_ignition_husky.launch.py` | Convenience wrapper: canonical launch file with Husky on, Parrot off. |
| `41068_ignition_parrot.launch.py` | Convenience wrapper: canonical launch file with Parrot on, Husky off. |
| `41068_navigation.launch.py` | Starts SLAM Toolbox and/or Nav2 for one robot, namespaced correctly. Included by the canonical launch file, once per enabled robot. |
| `41068_autonomy_demo.launch.py` | Standalone add-on: starts `basic_autonomy_demo.py` for a chosen robot (`robot:=husky1` or `robot:=parrot1`). Run in a second terminal after the main sim is up. |
| `41068_dynamic_world_demo.launch.py` | Standalone add-on: starts `dynamic_world_demo.py`. Run in a second terminal, only works with `world:=large_demo`. |

### `urdf_husky/`
| File | Role |
|---|---|
| `husky.urdf.xacro` | Full Husky robot description — real Clearpath Husky geometry, wheels, physics-enabled collision. |
| `husky.gazebo.xacro` | Gazebo plugins/sensors for the Husky (drive control, odometry, sensors). |
| `wheel.urdf.xacro` | Reusable wheel definition, included once per wheel. |
| `common_properties.urdf.xacro` | Shared material/colour definitions. |
| `meshes/*.dae` | 3D visual models (chassis, wheels, bumper). |

### `urdf_parrot/`
| File | Role |
|---|---|
| `parrot.urdf.xacro` | "Drone" body description — visual meshes, IMU link, lidar link (`base_scan`), camera link tilted 45° down. Collision geometry is present but commented out. |
| `parrot.gazebo.xacro` | The Parrot's actual simulated behaviour: `VelocityControl` plugin (moves the body directly from `cmd_vel`, which is why it "floats" rather than falling), `OdometryPublisher`, `JointStatePublisher`, plus the lidar and RGB-D camera sensor definitions. |
| `common_properties.urdf.xacro` | Shared material/colour definitions (same pattern as Husky). |
| `model.sdf` / `model.config` | A separate standalone Gazebo-model description of the Parrot (used for Fuel/thumbnail-style purposes) — **not** what's actually spawned; the live sim uses the xacro via the `robot_description` topic instead. |
| `meshes/*.dae` | 3D visual models (hull, 4 propellers). |
| `thumbnails/*.png` | Preview images, cosmetic only. |

### `worlds/`
| File | Role |
|---|---|
| `simple_trees.sdf` | Small, minimal world — good for quick iteration/testing. |
| `large_demo.sdf` | Bigger world; also contains four extra pre-placed demo models (`demo_animal`, `demo_tree_healthy`, `demo_tree_fire`, `demo_tree_burnt`) used by the dynamic-world example. |

### `models/`
| Folder | Role |
|---|---|
| `forest_plane/`, `forest_wall/`, `grass_plane/` | Reusable ground/vegetation props referenced by the world files (each has a `model.sdf` + `model.config` + texture files). |

### `config/`
| File | Role |
|---|---|
| `ignition_server.config` | Gazebo *server*-level plugin config (e.g. Sensors system), loaded globally via an environment variable so individual world files don't need to repeat it. |
| `gazebo_bridge_clock.yaml` | Bridges the simulation clock from Gazebo to ROS. |
| `gazebo_bridge_husky.yaml` / `gazebo_bridge_husky1.yaml` | Which Gazebo topics ↔ ROS topics are bridged for the Husky (odometry, IMU, lidar, camera, `cmd_vel`, etc). |
| `gazebo_bridge_parrot.yaml` / `gazebo_bridge_parrot1.yaml` | Same, for the Parrot. |
| `slam_params.yaml` / `slam_params_husky1.yaml` / `slam_params_parrot1.yaml` | Tuning parameters for SLAM Toolbox. |
| `nav2_params.yaml` / `nav2_params_husky1.yaml` / `nav2_params_parrot1.yaml` | Tuning parameters for Nav2 (planners, costmaps, controllers). |
| `robot_localization.yaml` / `robot_localization_husky1.yaml` / `robot_localization_parrot1.yaml` | EKF fusion settings (which sensors feed in, how they're weighted). |
| `41068.rviz`, `41068_husky1.rviz`, `41068_parrot1.rviz` | Saved RViz layouts. |

### `scripts/`
| File | Role |
|---|---|
| `basic_autonomy_demo.py` | Example autonomy node: reads the SLAM map + a crude camera-brightness signal, picks a random reachable point, sends it to Nav2, repeats. Meant to be replaced/extended, not used as-is — this is your template for "read sensors → decide → send a Nav2 goal." |
| `dynamic_world_demo.py` | Example world-manipulation node: moves the `large_demo.sdf` demo props around by calling Gazebo's `set_pose` service via the `gz`/`ign` command line tool. Template for "move/place things in the world from code." |

### Root files
| File | Role |
|---|---|
| `package.xml` | ROS package manifest — name, version, and dependencies (other ROS packages this one needs installed). |
| `CMakeLists.txt` | Build/install instructions — tells `colcon build` which folders and scripts to install so `ros2 launch`/`ros2 run` can find them. |
| `README.md` | Official setup/usage instructions from the teaching team. |

---

## Part 5 — Naming convention cheat sheet

For a robot in namespace `<robot>` (`husky1` or `parrot1`):

- ROS topics: `/<robot>/...` (e.g. `/husky1/scan`, `/parrot1/camera/image`)
- Gazebo model name: `<robot>` (e.g. `husky1`)
- Gazebo-side topics: `/model/<robot>/...`
- TF topics: `/<robot>/tf`, `/<robot>/tf_static`
- TF frame IDs: `<robot>_map`, `<robot>_odom`, `<robot>_base_link`, `<robot>_base_scan`, `<robot>_camera_link`

If you add a second Parrot later, following this same pattern (`parrot2`) is what
makes multi-robot "just work" without redesigning anything.

---

## Part 6 — Notes for the search-and-rescue drone project

- The perception/mapping/navigation pipeline (lidar, camera, SLAM, Nav2) already
  works for the Parrot out of the box — no changes needed to get it mapping.
- Person-detection and search-pattern logic don't exist yet — you'll write new
  nodes, using `basic_autonomy_demo.py` as a structural template (map + camera +
  TF + Nav2 action client).
- Vertical motion: the `VelocityControl` Gazebo plugin (in `parrot.gazebo.xacro`)
  sets the drone's velocity directly from `cmd_vel`, and nothing currently sends a
  nonzero `linear.z` — but the plugin itself isn't limited to 2D. Worth testing
  early by publishing a `Twist` with nonzero `z` to `/parrot1/cmd_vel`.
- "Dropping" the first aid kit can reuse the same Gazebo `set_pose` service pattern
  already demonstrated in `dynamic_world_demo.py`.
- Collision is currently disabled for the Parrot (commented out in
  `parrot.urdf.xacro`) — relevant if you want it to physically interact with
  anything (e.g. detect it's reached the ground, or avoid trees).

# Codebase Guide - 41068 Ignition Bringup

Internal reference written during development, to get the team up to speed on the
stack and to record how the trickier parts work. Parts 1-3 and 5 are introductory
notes on ROS 2 and this package's layout; **Parts 6-7 document project-specific
behaviour, including night mode**, and are the parts worth reading.

Come back to this any time you've forgotten what something does.

---

## Part 1 - ROS 2 concepts, explained simply

ROS 2 (Robot Operating System 2) is not an operating system - it's a framework for
getting many separate programs to talk to each other. Everything below is a concept
you'll see over and over in this codebase.

### Node
A **node** is one running program - e.g. "the SLAM node," "the camera driver node,"
"your autonomy script." A robot system is just lots of small nodes running at once
and talking to each other. `basic_autonomy_demo.py` is an example of a node.

### Topic
A **topic** is a named channel that nodes publish messages to, or subscribe to read
messages from. It's one-way, many-to-many, and continuous (a stream), like a radio
frequency. Example: `/husky1/scan` carries lidar data; any node can subscribe to it
without the publisher knowing or caring who's listening.

### Message type
Every topic carries one specific **message type** - a fixed data structure. E.g.
`sensor_msgs/msg/Image` is an image, `nav_msgs/msg/OccupancyGrid` is a 2D map,
`geometry_msgs/msg/Twist` is a velocity command (linear x/y/z + angular x/y/z).

### Service
A **service** is a request/response call - you ask a question, you get one answer
back, like a phone call. Used for occasional one-off asks rather than streams.
Gazebo's `set_pose` (used in `dynamic_world_demo.py`) is a service call.

### Action
An **action** is for long-running tasks with feedback and a final result - like
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
`/parrot1/map` respectively - same code, no clashes, and it's what makes running two
robots at once possible without rewriting anything.

### TF (transform tree)
**TF** answers "where is X relative to Y?" for every part of the robot and the
world, continuously, as things move. Each robot has a chain of coordinate frames:
`<robot>_map` → `<robot>_odom` → `<robot>_base_link` → sensor frames like
`<robot>_base_scan`, `<robot>_camera_link`. `basic_autonomy_demo.py`'s
`_lookup_robot_pose()` uses TF to find "where is the robot right now, in map
coordinates?"

- `map` frame: fixed to the world, corrected by SLAM (best long-term global estimate).
- `odom` frame: drifts slowly over time, but is smooth (no sudden jumps) - good for
  short-term motion.
- `base_link`: the robot's own body frame.

### Launch file
A **launch file** (`.launch.py`) is a Python script that starts a group of nodes
together with the right parameters, remappings, and conditions - instead of you
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

## Part 2 - File type glossary

| Extension | What it is |
|---|---|
| `.py` (in `launch/`) | A launch file - starts and configures a group of nodes. |
| `.py` (in `scripts/`) | An actual ROS node - a running program with logic. |
| `.urdf.xacro` | Robot description: links (rigid parts), joints (how parts connect/move), sensor placements, visual meshes. `xacro` is a macro language that generates plain URDF - it lets you use variables/reuse (e.g. `${prefix}` so the same file works for `husky1` or a future `husky2`). |
| `.gazebo.xacro` | Gazebo-specific additions to a robot description: physics plugins, sensor simulation parameters (noise, update rate), which aren't part of the "real robot" description. |
| `.sdf` | Simulation Description Format - Gazebo's native format for describing worlds (`worlds/*.sdf`) or standalone models (`models/*/model.sdf`). Different from URDF; URDF describes *robots* for ROS generally, SDF describes *everything* for Gazebo specifically. |
| `.yaml` | Plain structured config data (key: value), used here for: Nav2/SLAM/EKF tuning parameters, and topic-bridge definitions (which Gazebo topics map to which ROS topics). |
| `.rviz` | A saved layout/config for RViz (the 3D visualisation tool) - which displays are shown, their settings, camera view. |
| `model.config` | Metadata for a Gazebo model folder (name, version, path to the `.sdf`) - lets Gazebo find and load a model by name. |

---

## Part 3 - How a simulation actually boots up

Running, e.g.:
```
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true
```
triggers this chain:

1. **`41068_ignition_husky.launch.py`** - a thin wrapper. It just calls the file
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
launch code supports Husky-only, Parrot-only, or both at once - it's just whether
each robot's block of actions is enabled.

The two "add-on" launch files (`41068_autonomy_demo.launch.py`,
`41068_dynamic_world_demo.launch.py`) are separate and lightweight: they assume the
simulation above is already running, and just start one extra Python node each.

---

## Part 4 - File-by-file reference

### `launch/`
| File | Role |
|---|---|
| `41068_ignition.launch.py` | Canonical launch file - starts Gazebo + robots + (optionally) SLAM/Nav2/RViz. Everything else calls into this. |
| `41068_ignition_husky.launch.py` | Convenience wrapper: canonical launch file with Husky on, Parrot off. |
| `41068_ignition_parrot.launch.py` | Convenience wrapper: canonical launch file with Parrot on, Husky off. |
| `41068_navigation.launch.py` | Starts SLAM Toolbox and/or Nav2 for one robot, namespaced correctly. Included by the canonical launch file, once per enabled robot. |
| `41068_autonomy_demo.launch.py` | Standalone add-on: starts `basic_autonomy_demo.py` for a chosen robot (`robot:=husky1` or `robot:=parrot1`). Run in a second terminal after the main sim is up. |
| `41068_dynamic_world_demo.launch.py` | Standalone add-on: starts `dynamic_world_demo.py`. Run in a second terminal, only works with `world:=large_demo`. |

### `urdf_husky/`
| File | Role |
|---|---|
| `husky.urdf.xacro` | Full Husky robot description - real Clearpath Husky geometry, wheels, physics-enabled collision. |
| `husky.gazebo.xacro` | Gazebo plugins/sensors for the Husky (drive control, odometry, sensors). |
| `wheel.urdf.xacro` | Reusable wheel definition, included once per wheel. |
| `common_properties.urdf.xacro` | Shared material/colour definitions. |
| `meshes/*.dae` | 3D visual models (chassis, wheels, bumper). |

### `urdf_parrot/`
| File | Role |
|---|---|
| `parrot.urdf.xacro` | "Drone" body description - visual meshes, IMU link, lidar link (`base_scan`), camera link tilted 45° down. Collision geometry is present but commented out. |
| `parrot.gazebo.xacro` | The Parrot's actual simulated behaviour: `VelocityControl` plugin (moves the body directly from `cmd_vel`, which is why it "floats" rather than falling), `OdometryPublisher`, `JointStatePublisher`, plus the lidar and RGB-D camera sensor definitions. |
| `common_properties.urdf.xacro` | Shared material/colour definitions (same pattern as Husky). |
| `model.sdf` / `model.config` | A separate standalone Gazebo-model description of the Parrot (used for Fuel/thumbnail-style purposes) - **not** what's actually spawned; the live sim uses the xacro via the `robot_description` topic instead. |
| `meshes/*.dae` | 3D visual models (hull, 4 propellers). |
| `thumbnails/*.png` | Preview images, cosmetic only. |

### `worlds/`
| File | Role |
|---|---|
| `simple_trees.sdf` | Small, minimal world - good for quick iteration/testing. |
| `large_demo.sdf` | Bigger world: the Blue Mountains scenery, the AeroAid ute, `person1`, two animals beside `person1` (`demo_animal`, a kangaroo, and `wombat1`), and 12 more kangaroos and wombats scattered over the terrain by `scripts/place_animals.py`. |

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
| `basic_autonomy_demo.py` | Example autonomy node: reads the SLAM map + a crude camera-brightness signal, picks a random reachable point, sends it to Nav2, repeats. Meant to be replaced/extended, not used as-is - this is your template for "read sensors → decide → send a Nav2 goal." |
| `dynamic_world_demo.py` | Example world-manipulation node: moves the `large_demo.sdf` demo props around by calling Gazebo's `set_pose` service via the `gz`/`ign` command line tool. Template for "move/place things in the world from code." |

### Root files
| File | Role |
|---|---|
| `package.xml` | ROS package manifest - name, version, and dependencies (other ROS packages this one needs installed). |
| `CMakeLists.txt` | Build/install instructions - tells `colcon build` which folders and scripts to install so `ros2 launch`/`ros2 run` can find them. |
| `README.md` | Official setup/usage instructions from the teaching team. |

---

## Part 5 - Naming convention cheat sheet

For a robot in namespace `<robot>` (`husky1` or `parrot1`):

- ROS topics: `/<robot>/...` (e.g. `/husky1/scan`, `/parrot1/camera/image`)
- Gazebo model name: `<robot>` (e.g. `husky1`)
- Gazebo-side topics: `/model/<robot>/...`
- TF topics: `/<robot>/tf`, `/<robot>/tf_static`
- TF frame IDs: `<robot>_map`, `<robot>_odom`, `<robot>_base_link`, `<robot>_base_scan`, `<robot>_camera_link`

If you add a second Parrot later, following this same pattern (`parrot2`) is what
makes multi-robot "just work" without redesigning anything.

---

## Part 6 - Notes for the search-and-rescue drone project

- The perception/mapping/navigation pipeline (lidar, camera, SLAM, Nav2) already
  works for the Parrot out of the box - no changes needed to get it mapping.
- Person-detection and search-pattern logic don't exist yet - you'll write new
  nodes, using `basic_autonomy_demo.py` as a structural template (map + camera +
  TF + Nav2 action client).
- Vertical motion: the `VelocityControl` Gazebo plugin (in `parrot.gazebo.xacro`)
  sets the drone's velocity directly from `cmd_vel`, and nothing currently sends a
  nonzero `linear.z` - but the plugin itself isn't limited to 2D. Worth testing
  early by publishing a `Twist` with nonzero `z` to `/parrot1/cmd_vel`.
- "Dropping" the first aid kit can reuse the same Gazebo `set_pose` service pattern
  already demonstrated in `dynamic_world_demo.py`.
- Collision is currently disabled for the Parrot (commented out in
  `parrot.urdf.xacro`) - relevant if you want it to physically interact with
  anything (e.g. detect it's reached the ground, or avoid trees).

---

## Part 7 - Night mode

A dark world where only the thermal camera can see. Body heat holds the
search, a floodlight under the scout switches on, and the RGB (YOLO)
detector confirms the person in the lit area. Off by default; day runs skip
every night-only step.

```bash
LIBGL_ALWAYS_SOFTWARE=1 HEADLESS=true NIGHT=true ./run_blue_mountains.sh
```

### How it works

1. `run_blue_mountains.sh` writes the world Gazebo loads to
   `/tmp/aeroaid_large_demo.sdf` (a copy of `worlds/large_demo.sdf` with the
   casualty moved). With `NIGHT=true` it also `sed`s that copy: `<ambient>` to
   0.03, `<background>` black, and the sun's `<intensity>` to 0.05. It exits
   with an error if the sun edit doesn't apply. `worlds/large_demo.sdf` is
   never changed.
2. The script starts the ground station with `-p night_mode:=$NIGHT`.
3. The scout always has a spotlight named `flashlight`, pointing down from
   `base_link`, defined off (black, intensity 0) in `parrot.gazebo.xacro`
   inside the `cameras` block (so the courier has none).
4. `config/gazebo_bridge_parrot1.yaml` bridges ROS `/parrot1/flashlight`
   (`ros_gz_interfaces/msg/Light`) to Gazebo `/world/large_demo/light_config`.
   Gazebo's UserCommands system (loaded by `config/ignition_server.config`)
   applies each message to the light of that name.
5. `UI/ground_station.py`, `GroundStationNode`: `night_mode` picks the
   starting `flashlight_mode`, THERMAL at night and OFF by day. In THERMAL,
   `set_hold()` calls `_send_light(hold)`, so the light is on exactly while the
   search is held (thermal check running, contact waiting, or after a confirm
   until RESUME). A 1 s timer (`_resend_light`) re-sends the current state,
   because a message sent before the bridge connects is lost. A day run that
   never touches the light publishes nothing to it (`TestNightFloodlight` in
   `test/test_dismissed_objects.py`, `TestDefaultsUnchanged` in
   `test/test_settings.py`).
6. The operator can change the mode (OFF / ON / THERMAL) on the SETUP page,
   see "Operator settings" below.

Everything else is the normal contact workflow in
`MainWindow._check_detections`: heat starts a check and holds the search, a
YOLO human whose box sits on the heat raises the contact, and nothing seen
within `rgb_check_s` resumes the search.

### Rules that break it if changed

- **`_send_light` must match the `<light>` in the xacro.** Gazebo replaces the
  whole light with each message, so a field left at zero (`direction`,
  `range`, spot angles) changes the light. Only brightness (diffuse,
  specular, intensity) should differ between on and off.
- **The light name is `flashlight`**, both in the xacro and in `_send_light`.
- **`<visualize>false</visualize>` stays on the light.** Without it Gazebo
  draws the cone's outline into the scene and the scout's own camera shows it
  as bright green lines.
- **The bridge topic has the world name hard-coded** (`large_demo`). A
  different world needs its own `/world/<name>/light_config`.
- **The thermal and RGB cameras share mount, FOV (1.4 rad) and resolution
  (720x480).** `box_on_heat`, `box_region` and the frame-edge rule compare
  pixels between the two images directly. If the cameras ever differ,
  compare ground positions (`pixel_to_ground`) instead.

### Related detection rules (not night-only)

These run in day mode too. They were added after night-mode testing showed
the person being missed:

- `_detect_person` drops hot blobs touching the frame edge (`EDGE_MARGIN`,
  5%, same as the detector's `edge_margin_frac`). Otherwise heat from a body
  just entering the frame holds the search with the person at the edge,
  where YOLO marks them "entering" and never confirms.
- `fresh_animal_only` ends a check early for a wombat or kangaroo only if
  the heat outside the animals' boxes is under `person_min_pixels // 4`.
  Otherwise a person standing beside an animal is missed.

### Tuning

| What | Where |
|---|---|
| How dark | the `sed` values in `run_blue_mountains.sh` (`NIGHT` block) |
| Light brightness, cone, range | `_send_light` and the xacro `<light>` (change both) |
| How long the light has to confirm | `rgb_check_s` ground station parameter (6 s) |

### Operator settings (SETUP page)

Page 5 of the ground station (`SETUP`, key `5`). The values change with the
right column (F1-F5) and apply at once. Each also has a ground station
parameter for where it starts.

| Setting | Buttons | Parameter (default) | Effect |
|---|---|---|---|
| Flashlight | LIGHT cycles OFF → ON → THERMAL | `flashlight_mode` (empty: THERMAL at night, OFF by day) | THERMAL = on while the search holds |
| Minimum body heat | HEAT+ / HEAT- (0.5 K) | `person_min_k` (305 K) | thermal pixels this warm count as body heat; kept inside `thermal_min_k`-`thermal_max_k` |
| Thermal detection | IR DET | `thermal_detection` (true) | off: a YOLO human raises a contact without heat |
| YOLO detection | AI DET | `rgb_detection` (true) | off: body heat raises a contact on its own, at the heat's ground position (`thermal_hit_xy`); animals too |

At least one detector stays on. Switching one off mid-check ends the check.
The status strip shows `AI OFF` / `IR OFF` while one is off. The logic is in
the `set_*` methods of `GroundStationNode` and in `_check_detections`; tests
in `test/test_settings.py`.

### Debugging

- `[UserCommands.cc] Failed to find light with name [flashlight]` in the sim
  window: the scout has not spawned (no scout, no light), or the light name
  doesn't match. Check for `[spawn_parrot1]: OK creation of entity.`
- Is the light in Gazebo, and what are its values?
  ```bash
  ign service -s /world/large_demo/scene/info --reqtype ignition.msgs.Empty \
    --reptype ignition.msgs.Scene --timeout 15000 --req '' | grep -A30 'name: "flashlight"'
  ```
- Switch it by hand. The reply `data: true` means Gazebo accepted it. The
  ground station re-sends its own state every second, so it will undo this
  unless the search is held.
  ```bash
  ign service -s /world/large_demo/light_config --reqtype ignition.msgs.Light \
    --reptype ignition.msgs.Boolean --timeout 5000 --req 'name: "flashlight", type: SPOT,
    pose: {position: {z: -0.15}}, diffuse: {r: 1, g: 1, b: 1, a: 1}, specular: {r: 0.3,
    g: 0.3, b: 0.3, a: 1}, attenuation_constant: 0.2, attenuation_linear: 0.01, range: 40,
    direction: {z: -1}, spot_inner_angle: 1.3, spot_outer_angle: 1.6, spot_falloff: 1, intensity: 3'
  ```
  On the live sim, mean RGB brightness went from about 7 (light off) to about
  100 (light on).

### Reverting

Remove all of these together:

1. `run_blue_mountains.sh`: the `NIGHT=` setting, the `if [[ "$NIGHT" == true ]]`
   block, and `-p night_mode:=$NIGHT` on the ground-station line.
2. `urdf_parrot/parrot.gazebo.xacro`: the `<gazebo reference="${prefix}base_link">`
   block holding `<light name="flashlight">`.
3. `config/gazebo_bridge_parrot1.yaml`: the `flashlight` entry.
4. `UI/ground_station.py`: the `night_mode` and `flashlight_mode` parameters,
   `light_pub` and timer in `__init__`, the `_send_light` call in `set_hold`,
   `light_on`, `_resend_light`, `set_flashlight_mode`, `_send_light` itself,
   the LIGHT button on the SETUP page, and the `ros_gz_interfaces.msg.Light`
   import.
5. `test/test_dismissed_objects.py`: `TestNightFloodlight`, and
   `test/test_settings.py`: `TestFlashlight`.

Leave the related detection rules above unless they are what's wrong; they
fix day-mode misses too. Rebuild with `colcon build --symlink-install`.

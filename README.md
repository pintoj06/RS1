# AeroAid: autonomous search and rescue in remote bushland

**41068 Robotics Studio 1, Team 4**
Harry Mentis · Jacob Singh · Jayden Pinto · John McGuire · William Lowing

A two-drone search-and-rescue system simulated in Ignition Gazebo Fortress on ROS 2 Humble.
An operator draws a search area on a map. A **scout** drone flies a camera-derived lawnmower
sweep over it, finds a casualty using fused thermal and RGB detection, and holds for the
operator to rule on the sighting. Once confirmed, a **courier** drone flies out and drops a
first-aid package, then both drones return to base.

The world is a model of the Three Sisters in the Blue Mountains, NSW, built in Blender with
per-object thermal values.

> This repository began as the subject's `41068_ignition_bringup` package. The original
> documentation is retained at the bottom of this file under
> **Original package documentation**. Everything above that heading describes what Team 4 built.

---

## Quick start

Install the dependencies first (see **Installing the dependencies** below), then:

```bash
# 1. get the code into a colcon workspace
mkdir -p ~/41068_ws/src
git clone https://github.com/pintoj06/RS1 ~/41068_ws/src/rs1

# 2. build
source /opt/ros/humble/setup.bash
cd ~/41068_ws
colcon build --symlink-install
source install/setup.bash

# 3. run
cd ~/41068_ws/src/rs1
./run_blue_mountains.sh

# to stop everything, including strays
./run_blue_mountains.sh kill
```

Keep the repository directly inside the workspace's `src` folder. The run script finds the
workspace two folders above itself.

`run_blue_mountains.sh` is the **only** entry point you need. It opens five terminal tabs:

| Tab | Starts |
|---|---|
| 1/5 | Simulation (Blue Mountains world) + scout + courier + Nav2 + SLAM |
| 2/5 | Ground station (the operator GUI) |
| 3/5 | Scout RGB person detector (YOLO) |
| 4/5 | Scout search node |
| 5/5 | Courier node |

## What you should see

- A **Gazebo** window showing the Three Sisters terrain, two drones on a ute, and scattered wildlife.
- The **ground station** window: a map view with the aerial photo and the live SLAM map, a camera
  feed, and soft buttons down each side.
- In the ground station's status bar, **`AI OK`** and **`LINK OK`**.
  If `AI ---` shows in red the YOLO detector has not started. See **Troubleshooting**.
  **Nothing can raise a contact without it.**

## Operating it

1. Press **`LAUNCH`**. Both drones climb off the ute: the scout to 10 m, the courier to 9 m.
2. Press **`SET AREA`**, then drag a box on the map to define the search area.
   (Either order works - an area drawn before launch is held and used once the scout is airborne.)
3. The scout plans a lawnmower sweep and begins flying it. The planned route is drawn in green,
   and ground the camera has covered is shaded cyan behind it.
4. Ground in the search area that the scout has not settled is shown in two other ways: **amber**
   means it could not yet be confirmed reachable, and **grey** means obstacle or walled off with
   no way in. Reachable ground still to be swept is shaded only faintly.
5. When body heat and a YOLO human detection coincide, the search **holds** and a contact is
   raised. Open the contacts page and press **`CONFIRM`** or **`DISMISS`**.
   `DISMISS` ignores the contact and the sweep carries on.
6. On `CONFIRM`, the courier flies to the position, drops the package and returns. The scout then
   flies home and reports the search complete.
7. **`ESTOP`** (or the spacebar) stops both drones immediately. **`RESUME`** releases it.

## Scenario switches

Set as environment variables before the run script:

```bash
PERSON_SPOT=3 ./run_blue_mountains.sh                 # fixed casualty position (0-3), not random
NIGHT=true ./run_blue_mountains.sh                    # night search; the scout's floodlight is used
HEADLESS=true ./run_blue_mountains.sh                 # no Gazebo window, runs faster
LARGE_COURIER=false ./run_blue_mountains.sh           # standard-size courier
RETURN_AFTER_DELIVERY=false ./run_blue_mountains.sh   # courier stays at the contact
SIM_WAIT=45 ./run_blue_mountains.sh                   # longer wait for Gazebo on a slow machine
```

There are **two casualties**: one always at world (-6.36, -3.07) beside a kangaroo and a wombat,
a deliberate false-positive test, and a second at one of four preset spots, chosen at random each
run so the search is not always the same problem.

## Installing the dependencies

Tested on Ubuntu 22.04 with ROS 2 Humble and Ignition Gazebo Fortress. Other versions are
untested. Install ROS 2 Humble and Gazebo Fortress as described in **Original package
documentation** below, then the extra ROS and Python packages:

```bash
sudo apt install ros-dev-tools \
    ros-humble-ros-ign ros-humble-ros-ign-interfaces ros-humble-ros-gz \
    ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
    ros-humble-robot-localization ros-humble-robot-state-publisher ros-humble-xacro \
    ros-humble-cv-bridge ros-humble-vision-msgs \
    python3-numpy python3-scipy python3-pyqt5 python3-pip
```

The YOLO detector needs some extra Python packages:

```bash
python3 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user ultralytics "opencv-python<5" "numpy<1.25"
python3 -m pip uninstall -y opencv-python
python3 -m pip install --user "opencv-python-headless<5" "numpy<1.25"
```

**All four commands matter.** Three traps, each of which we hit:

- **`numpy` must stay below 1.25.** `ultralytics` requires 1.23 or later and pulls numpy 2 or later
  by default, but the system SciPy supports only versions below 1.25, and SciPy is what the search
  node's flood fills run on. **1.24.4 satisfies both.** numpy 2 or later additionally breaks
  `cv_bridge`, which is compiled against numpy 1.x.
- **Use the headless OpenCV build.** `opencv-python` bundles its own Qt plugins, which replace the
  ones PyQt5 needs. The ground station then dies with
  `Could not load the Qt platform plugin "xcb"`.
- **Take torch and torchvision from the CPU index.** Otherwise `ultralytics` pulls a CUDA
  torchvision against CPU torch and the first camera frame fails with
  `operator torchvision::nms does not exist`. Check with `pip list --user | grep torch`. Both
  should end in `+cpu`.

Model weights (`weights/best.pt`) are committed, so no training is required.

### Running the tests

The tests need ROS sourced (`source install/setup.bash`) but no simulation:

```bash
cd ~/41068_ws/src/rs1 && python3 -m pytest test/ -q
```

440 tests, covering sweep planning, reachability, costmap settings, E-Stop, edge checks, the
courier route and return to base.

## Where the code is

| Path | What it is | Status |
|---|---|---|
| `scripts/search_node.py` | Scout decision making: sweep planning, reachability, vantage selection, replanning | custom |
| `UI/ground_station.py` | Operator GUI, thermal detection, contact logic, E-Stop | custom |
| `scripts/yolo_person_detector.py` | RGB human and animal detection | custom (uses `ultralytics`) |
| `scripts/courier_node.py` | Delivery: frame conversion, navigation, package release | custom |
| `scripts/takeoff_node.py` | Vertical flight, since Nav2 plans only in 2D | custom |
| `scripts/place_animals.py` | Scatters wildlife as false-positive distractors | custom |
| `worlds/large_demo.sdf`, `models/blue_mountains/` | Three Sisters terrain with thermal values | created |
| `blue_mountains_gazebo/` | Terrain and animal generation pipeline | created |
| `urdf_parrot/` | Parrot model plus thermal camera, floodlight, care package | adapted |
| `config/nav2_params_parrot*.yaml` | Nav2 planner, controller and costmaps per drone | adapted |
| `launch/` | `41068_search.launch.py` (scout), `41068_courier_demo.launch.py` (courier) | adapted |
| `test/` | 440 automated tests, no simulation required | custom |
| `CODEBASE_GUIDE.md` | Internal development reference; see Part 7 for night mode | custom |

## Key parameters

Search behaviour, in `scripts/search_node.py`, overridable as ROS parameters:

| Parameter | Default | Effect |
|---|---|---|
| `flight_altitude` | 10.0 m | Sets the camera footprint, and so the leg spacing |
| `overlap` | 0.8 | Leg spacing as a fraction of footprint width (about 9 m at 10 m altitude) |
| `obstacle_cost_threshold` | 77 | Costmap cost at which a cell blocks planning |
| `sight_block_threshold` | 99 | Costmap cost at which a cell blocks the camera's view |
| `map_review_period` | 2.0 s | How often reachability is re-evaluated |
| `vantage_patience` | 5 | Fruitless look-around trips before the scout gives up |
| `use_slam_walls` | true | Use SLAM's finer map for the walled-in test |

Detection, in `UI/ground_station.py`, also adjustable live on the **SETUP** page:

| Parameter | Default | Effect |
|---|---|---|
| `person_min_k` | 305.0 K | Temperature at which a pixel counts as body heat |
| `person_min_pixels` | 600 | Hot pixels needed before the search holds for a check |
| `rgb_check_s` | 6.0 s | How long the scout holds waiting for YOLO to confirm |

Navigation, in `config/nav2_params_parrot1.yaml`:

| Parameter | Default | Effect |
|---|---|---|
| `obstacle_max_range` | 39.0 m | **Must stay below the lidar's 40 m range**. See the comment in the file |
| `use_rotate_to_heading` | true | Pivot on the spot at sweep corners, then fly straight |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `AI ---` in red; no contact is ever raised | The YOLO detector did not start. Check tab 3/5 for a traceback, usually a missing or wrong-versioned `ultralytics`, `numpy` or `torch`. See **Installing the dependencies**. |
| `Could not load the Qt platform plugin "xcb"` | `opencv-python` is installed instead of `opencv-python-headless`. Rerun the last two install commands. |
| `operator torchvision::nms does not exist` on the first camera frame | torch and torchvision are from different builds. Both must end in `+cpu`. |
| A SciPy warning about the numpy version | numpy is 1.25 or later. Install `numpy==1.24.4`. |
| Changed a parameter and nothing happened | Node scripts are symlinked by `--symlink-install`, so a node restart is enough, but **YAML config changes need a full relaunch**, because Nav2 reads its parameters at startup. |
| Gazebo has not finished loading before the nodes start | `SIM_WAIT=45 ./run_blue_mountains.sh` |
| Heavy world, low frame rate | `HEADLESS=true`, and a longer `SIM_WAIT` |
| Leftover processes after a crash | `./run_blue_mountains.sh kill` |

## Known limitations

- The YOLO detector runs on the CPU at a few frames per second, which is why the scout has to hold
  while it confirms a contact.
- Nav2 plans in 2D at a fixed altitude, so a drone cannot fly over terrain it could in practice
  clear.
- See Section 5.3 of the final report for the full list.

---

# Original package documentation

*Everything below is the documentation supplied with the original `41068_ignition_bringup`
package (Husky, dynamic world and basic autonomy demos). It is retained because the installation
and launch instructions for the underlying simulation still apply.*

# 41068 Ignition Bringup

Bringup for *41068 Robotics Studio I*. This package launches a Husky UGV and/or a simple "Parrot" drone in custom Ignition Gazebo simulation worlds with trees and grass. We use **ROS 2 Humble** and **Ignition Gazebo Fortress**.

The aim of this package is to provide a starting point for your project, so that you have a simulation environment and basic autonomy working from the beginning. You are welcome to and encouraged to use and adapt this package in any way you see fit for your project. Get creative!

A single launch file is used for Husky-only, Parrot-only, and multi-robot simulations. This keeps the ROS namespaces, TF frames, SLAM, Nav2, Gazebo bridges, and RViz setup consistent across all modes. I strongly suggest working with a single-robot setup to begin with. The following instructions explain how to set up the different options.

Worlds contain resources found in [Gazebo Fuel](https://app.gazebosim.org/fuel/models). You might like to add other resources from Gazebo Fuel to your customised worlds.

This README details:
* How to install the package
* How to launch the simulation with a Husky ground robot
* How to launch the simulation with a Parrot aerial robot
* How to launch the simulation with both the Husky and Parrot robots
* How to run an demo involving dynamic simulation objects
* How to launch a basic autonomy script for the Husky and Parrot that reads in a map and camera, and moves to random locations in the world

## Installation

### Dependencies

First install the dependencies:

* If you haven't already, install ROS 2 Humble. Follow the instructions here: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
* Install Gazebo:
  ```bash
  sudo apt-get update && sudo apt-get install wget
  sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
  wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
  sudo apt-get update && sudo apt-get install ignition-fortress
  ```
* Install development tools, Gazebo bridges, Nav2, SLAM Toolbox, and robot localisation:
  ```bash
  sudo apt install ros-dev-tools ros-humble-robot-localization
  sudo apt install ros-humble-ros-ign ros-humble-ros-ign-interfaces
  sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox
  sudo apt install ros-humble-turtlebot4-simulator ros-humble-irobot-create-nodes
  sudo apt install python3-numpy
  ```
* Make sure that your installation is up to date. This is particularly important if you installed ROS a long time ago, such as in another subject. If you get errors here, resolve these before continuing.
  ```bash
  sudo apt update
  sudo apt upgrade
  ```

### This package

Now install this package:

* Create a new colcon workspace:
  ```bash
  mkdir -p ~/41068_ws/src
  ```
* Copy this package to the `src` directory in this workspace.
* Build the package. If you get an error suggesting a missing dependency, check that you have followed the installation instructions above.
  ```bash
  source /opt/ros/humble/setup.bash
  cd ~/41068_ws
  colcon build --symlink-install
  ```
* Source the workspace. If you add this to your `~/.bashrc`, you do not need to do this each time.
  ```bash
  source ~/41068_ws/install/setup.bash
  ```

### Important classroom network setting

By default, ROS 2 can automatically discover and subscribe to ROS topics from other computers on the same network. This is normally useful because it allows different robots and computers to communicate. However, in a classroom where everyone is on the same Wi-Fi network, it can cause confusing behaviour. For example, RViz might accidentally show the output of someone else's code instead of your own simulation.

To keep your ROS 2 system local to your own computer, set this environment variable:

```bash
export ROS_LOCALHOST_ONLY=1
```

To make this happen automatically in every new terminal, add it to your `~/.bashrc` file:

```bash
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
source ~/.bashrc
```

If you later want ROS 2 to communicate with other computers or real robots over the network, remove or comment out that line from your `~/.bashrc` and open a new terminal.

## Launch files

The canonical launch file is:

```bash
41068_ignition.launch.py
```

For the staged exercises, use these convenience wrappers:

```bash
41068_ignition_husky.launch.py
41068_ignition_parrot.launch.py
```

These wrappers simply call the canonical launch file with the appropriate robot enabled. This keeps the Husky-only, Parrot-only, and multi-robot setups consistent.

There are also two small add-on example launch files:

```bash
41068_dynamic_world_demo.launch.py
41068_autonomy_demo.launch.py
```

These add-on launch files are intended to be run from a second terminal after the main simulation is already running.

Important arguments:

* `husky:=true/false` - launch the Husky UGV. Default: `true`.
* `parrot:=true/false` - launch the Parrot drone. Default: `false`.
* `slam:=true/false` - launch SLAM Toolbox for each enabled robot. Default: `false`.
* `nav2:=true/false` - launch Nav2 for each enabled robot. Nav2 also starts SLAM. Default: `false`.
* `rviz:=true/false` - launch RViz. If multiple robots are enabled, one RViz window is opened for each robot. Default: `false`.
* `world:=simple_trees/large_demo` - choose the Gazebo world. Default: `simple_trees`.

Robots are namespaced even in single-robot mode:

* Husky topics are under `/husky1/...`; frames include `husky1_map`, `husky1_odom`, `husky1_base_link`, and `husky1_base_scan`.
* Parrot topics are under `/parrot1/...`; frames include `parrot1_map`, `parrot1_odom`, `parrot1_base_link`, and `parrot1_base_scan`.

This consistency makes it easier to move from one robot to two robots without changing your code structure.

## Stage 1: Launch the Husky UGV

Start with the Husky only:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py
```

Launch the Husky with SLAM, Nav2, and RViz:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true
```

Launch in the larger world:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true world:=large_demo
```

When launching with RViz, use the "Nav2 Goal" / "2D Goal Pose" tool to send a waypoint to the robot. The robot is navigating using Nav2. If it gets stuck, try the buttons in the Navigation 2 panel in the top right of RViz.

You can also drive the Husky using keyboard teleoperation from a separate terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/husky1/cmd_vel
```

## Stage 2: Launch the Parrot drone

The Parrot drone is intentionally simple. To keep it consistent with the Husky model, this drone is essentially a Husky-like robot floating near the ground, with gravity disabled. It does **not** accurately model drone flight dynamics. You are welcome to develop a more realistic drone model as part of your project. It currently only flies in 2D, but you are welcome to extend it to 3D flight.

Launch the Parrot only with SLAM, Nav2, and RViz:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_parrot.launch.py slam:=true nav2:=true rviz:=true world:=large_demo
```

The drones start on their landing pads on the AeroAid ute (`models/aeroaid_ute`) and stay there until LAUNCH is pressed in the ground station. Then `scripts/takeoff_node.py` climbs each one to a fixed flight altitude, and SLAM and Nav2 start once it is up. To have them climb straight away without the ground station, publish the launch yourself: `ros2 topic pub --once --qos-durability transient_local /aeroaid/launch std_msgs/msg/Bool "{data: true}"`. You can change that altitude in `launch/41068_ignition.launch.py` by finding the Parrot `add_robot(...)` block and changing the `takeoff_altitude` argument (keep `flight_altitude` in `41068_search.launch.py` in step for the scout).

The drone camera is set up to tilt downwards towards the ground. You can adjust this in `urdf_parrot/parrot.urdf.xacro` by finding the `camera_joint` and changing its pitch.

Since the drone has a very hard time navigating through the leaves of the trees, collisions are disabled for the drone. This lets it fly through objects. You can enable collisions by uncommenting the `collision` field in `urdf_parrot/parrot.urdf.xacro`, but note that the default navigation stack does not work well in that situation. You would need to further develop the collision avoidance planners for this challenge.

You can drive the Parrot using keyboard teleoperation from a separate terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/parrot1/cmd_vel
```

## Stage 3: Multi-robot challenge

Important: focus on a single-robot system first. Simulating multiple robots simultaneously introduces extra complexity in namespaces, TF, mapping, Nav2, and debugging. It may also significantly slow down the simulation, especially on laptops or older computers. If the multi-robot launch is too slow on your computer, stick with the single-robot version.

When you are ready, launch both robots together:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition.launch.py husky:=true parrot:=true slam:=true nav2:=true rviz:=true world:=large_demo
```

You should see both the Husky and Parrot appear in Gazebo. Two RViz windows will open: one for `/husky1` and one for `/parrot1`. Each RViz window is connected to the matching map, TF tree, robot model, and Nav2 instance, so you can command each robot separately.

The multi-robot launch intentionally uses the canonical launch file directly because it enables both robot instances at once.

## Dynamic world demo

This package includes a small example of changing the Gazebo world from a Python ROS node. It is intended as a simple starting point and source of ideas, not as a complete project solution.

The dynamic world demo is launched separately from the main simulation. First, start the normal simulation in one terminal. For example:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true world:=large_demo
```

Then, in a second terminal, start the dynamic world demo node:

```bash
ros2 launch 41068_ignition_bringup 41068_dynamic_world_demo.launch.py
```

This second launch file does not start Gazebo, robots, SLAM, Nav2, or RViz. It only starts the dynamic world Python node and connects to the already-running `large_demo` Gazebo world.

The `large_demo` world already contains a few simple demo models for this example. The add-on node moves these existing models:

* a small animal marker that slowly wanders around the forest floor; and
* a simple tree marker that cycles between healthy, fire, and burnt visual states.

The Python script is here:

```text
scripts/dynamic_world_demo.py
```

The example is deliberately simple. The demo models are defined in `worlds/large_demo.sdf`, and the Python script manipulates them by calling Gazebo's `set_pose` service. This keeps the add-on script focused on the dynamic behaviour rather than model creation.

The same add-on launch can be used while running the Husky, the Parrot, or both robots, as long as the main simulation is using `world:=large_demo`.

## Basic autonomy demo

This package also includes a small Python autonomy example. It is intended as skeleton code showing how a ROS node can read the current map, read a camera image, look up TF, send a goal to Nav2, and wait for the Nav2 action result.

First, start a robot with SLAM and Nav2. For example:

```bash
ros2 launch 41068_ignition_bringup 41068_ignition_husky.launch.py slam:=true nav2:=true rviz:=true world:=large_demo
```

Then, in a second terminal, run the autonomy demo for that robot:

```bash
ros2 launch 41068_ignition_bringup 41068_autonomy_demo.launch.py robot:=husky1
```

For the Parrot, first launch the Parrot simulation, then run:

```bash
ros2 launch 41068_ignition_bringup 41068_autonomy_demo.launch.py robot:=parrot1
```

The launch file only starts the autonomy node. It does not start Gazebo, SLAM, Nav2, or RViz. The example node is here:

```text
scripts/basic_autonomy_demo.py
```

This code is provided as a starting point for your project to develop your autonomy software. It currently performs a simple map-aware random walk, with a small camera-brightness score influencing the next Nav2 goal. Read the Python code and comments rather than treating this README as the main explanation. The code is deliberately simple so that you can adapt it for your own project. You are welcome to use it how you see fit, or you could disregard it and find a different starting point. While this example is in Python, you are also very welcome to use C++ instead.

## Notes on namespaces and TF

This package uses the following convention:

* ROS topics are namespaced: `/husky1/...`, `/parrot1/...`.
* Gazebo model names are `husky1` and `parrot1`.
* Gazebo model topics use `/model/husky1/...` and `/model/parrot1/...`.
* TF topics are namespaced: `/husky1/tf`, `/husky1/tf_static`, `/parrot1/tf`, `/parrot1/tf_static`.
* TF frame IDs are prefixed: `husky1_base_link`, `husky1_odom`, `husky1_map`, `parrot1_base_link`, `parrot1_odom`, `parrot1_map`.

Your code should use these namespaced topics and frames when interacting with the robots. The `basic_autonomy_demo.py` shows examples of how to use this.

## Errors

If you get errors, first check that you are following the instructions correctly. Otherwise, read the error messages carefully, search for the specific error, and discuss it with your team or the teaching staff.

### Jump back in time

If you continuously get an error like:

```bash
Detected jump back in time. Clearing TF buffer
```

and you see things flashing in RViz, this is probably due to the simulation clock being reset. This can happen if multiple Gazebo instances are running, perhaps because a previous Gazebo process crashed and did not close properly.

To fix this, try closing all terminals running the simulation and restarting the computer. You can also check for stale Gazebo processes with:

```bash
ps aux | grep -E "ign gazebo|gz sim|ros_gz|ros_ign" | grep -v grep
```

### Ogre Exception

If you get an error like:

```bash
[Ogre2RenderEngine.cc:989]  Unable to create the rendering window: OGRE EXCEPTION(3:RenderingAPIException): currentGLContext was specified with no current GL context in GLXWindow::create at /build/ogre-next-UFfg83/ogre-next-2.2.5+dfsg3/RenderSystems/GL3Plus/src/windowing/GLX/OgreGLXWindow.cpp (line 163)
```

[this thread](https://robotics.stackexchange.com/questions/111547/gazebo-crashes-immediately-segmentation-fault-address-not-mapped-to-object-0) suggests setting a bash variable before launching Gazebo:

```bash
export QT_QPA_PLATFORM=xcb
```

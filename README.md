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

* `husky:=true/false` — launch the Husky UGV. Default: `true`.
* `parrot:=true/false` — launch the Parrot drone. Default: `false`.
* `slam:=true/false` — launch SLAM Toolbox for each enabled robot. Default: `false`.
* `nav2:=true/false` — launch Nav2 for each enabled robot. Nav2 also starts SLAM. Default: `false`.
* `rviz:=true/false` — launch RViz. If multiple robots are enabled, one RViz window is opened for each robot. Default: `false`.
* `world:=simple_trees/large_demo` — choose the Gazebo world. Default: `simple_trees`.

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

The drone currently flies at a fixed altitude. You can change its spawn height in `launch/41068_ignition.launch.py` by finding the Parrot `add_robot(...)` block and changing the `z` argument.

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

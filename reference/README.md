# Reference / archive

Files in this folder are **not built and not installed**. They are kept only as
a reference to look at if useful later.

These were AI-generated during early exploration of the search-planner idea and
are being rewritten by hand instead. Nothing in the package depends on them.

| File | What it was |
|---|---|
| `search_manager_node.py` | First draft of the scout drone's search-grid sweep node: builds a boustrophedon ("lawnmower") pattern over a rectangle and sends each waypoint to Nav2's `NavigateToPose` action. Also published RViz markers for the search area. |
| `41068_search_demo.launch.py` | Launch file that started the above node inside the `/parrot1` namespace, with the search rectangle passed as launch arguments. |

## To restore one of these

1. Move the file back (`scripts/` for nodes, `launch/` for launch files).
2. If it is a node, add it to the `install(PROGRAMS ...)` list in `CMakeLists.txt`.
3. `colcon build --symlink-install` from the workspace root.

## Note on the bug hunt these were involved in

Time was lost chasing Nav2 goals that kept aborting with `status 6` (ABORTED).
The cause turned out to be **localisation**, not the planning code: SLAM's
`map -> odom` transform was at one point off by ~6 m and 57 degrees, so Nav2
believed the drone was somewhere it was not.

Useful check before trusting any navigation run:

```bash
timeout 8 ros2 run tf2_ros tf2_echo parrot1_map parrot1_odom \
  --ros-args -r /tf:=/parrot1/tf -r /tf_static:=/parrot1/tf_static
```

Near-zero translation and rotation means localisation is healthy. Large values
mean nothing built on top of Nav2 will work until that is fixed. The
`large_demo` world gives the lidar far more features than `simple_trees`
(which has only two trees), so it drifts less.

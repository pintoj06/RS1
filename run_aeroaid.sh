#!/usr/bin/env bash
# Start the whole AeroAid stack in separate terminal tabs.
#
#   ./run_aeroaid.sh          start everything
#   ./run_aeroaid.sh kill     stop everything and clean up strays
#
# Separate tabs rather than one combined launch file on purpose: when
# something breaks you need to read one node's log without four others
# scrolling past it, and that is most of what debugging this stack is.

set -u

WS=~/ros2_ws
PKG=41068_ignition_bringup
SCOUT=parrot1
COURIER=parrot2

# Every node that does a TF lookup needs these: this package pushes TF into
# each robot's namespace instead of the global /tf.
TF_REMAP="-r /tf:=/${SCOUT}/tf -r /tf_static:=/${SCOUT}/tf_static"

cleanup() {
    echo "Stopping everything..."
    pkill -f 'ign gazebo'      2>/dev/null
    pkill -f parameter_bridge  2>/dev/null
    pkill -f ruby              2>/dev/null
    pkill -f ground_station    2>/dev/null
    pkill -f search_node       2>/dev/null
    pkill -f courier_node      2>/dev/null
    pkill -f rviz2             2>/dev/null
    sleep 1
    ros2 daemon stop >/dev/null 2>&1
    ros2 daemon start >/dev/null 2>&1
    echo "Done."
}

if [[ "${1:-}" == "kill" ]]; then
    cleanup
    exit 0
fi

# Always start from a clean slate. A stale Gazebo from the last run will
# happily keep publishing sensor topics while the new launch fails around it,
# which looks like a code problem and is not one.
cleanup

if [[ ! -f "$WS/install/setup.bash" ]]; then
    echo "No workspace at $WS/install - run colcon build first." >&2
    exit 1
fi

# Open each stage in its own tab. 'exec bash' keeps the tab alive after the
# node exits, so you can still read why it died.
tab() {
    local title="$1"; shift
    gnome-terminal --tab --title="$title" -- \
        bash -c "source /opt/ros/humble/setup.bash;
                 source $WS/install/setup.bash;
                 $*;
                 echo; echo '--- $title exited ---';
                 exec bash"
}

echo "1/4  Simulation + Nav2 + RViz"
tab "sim" "ros2 launch $PKG 41068_ignition.launch.py \
    parrot:=true courier:=true large_courier:=true husky:=false \
    nav2:=true rviz:=true"

# Gazebo, the bridges, SLAM and Nav2 all have to be up before anything tries
# to plan. 20s is generous on a slow VM; drop it if yours is quicker.
echo "     waiting 20s for the simulation to come up..."
sleep 20

echo "2/4  Ground station"
tab "gui" "ros2 run $PKG ground_station.py --ros-args \
    -r __ns:=/$SCOUT -p robot_name:=$SCOUT $TF_REMAP"

sleep 2

echo "3/4  Scout search"
tab "scout" "ros2 launch $PKG 41068_search.launch.py robot:=$SCOUT"

sleep 2

echo "4/4  Courier"
tab "courier" "ros2 launch $PKG 41068_courier_demo.launch.py robot:=$COURIER"

cat <<EOF

All four tabs started.

  Check it worked:   ros2 node list
  Stop everything:   $0 kill

If the map stays empty, check the lidar is returning something:
  ros2 topic echo /$SCOUT/scan --field ranges --once | head -3
EOF